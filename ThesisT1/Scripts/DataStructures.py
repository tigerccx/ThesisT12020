import os

import numpy as np
from skimage import transform
import torch.utils.data as data
import cv2

import nibabel as nib

from Utils import NiiProcessor
from Utils import ImageProcessor
from Utils import CommonUtil

#MACRO
DEBUG=False

# Use index-last ndarray for storing data
# Use Grey1 img
class ImgDataSet(data.Dataset):
    # Accept array of niis
    def __init__(self, niisData, niisMask, slices=1, classes=2, resize=None, aug=None, preproc=None):
        # TODO: Maybe check data-mask match
        self.imgDataWrappers = np.empty(0)  # [ImgDataWrapper, ...]

        print("Constructing Data and Masks...")

        for i in range(len(niisData)):
            niiImg = niisData[i]
            niiMask = niisMask[i]

            atlasImg = NiiProcessor.ReadGrey1ImgsFromNii(niiImg)
            atlasMask = NiiProcessor.ReadGreyStepImgsFromNii(niiMask)

            atlasesImg = None
            atlasesMask = None

            if aug is not None:
                '''
                aug: function for data augmentation
                In: ndarray[H,W,Slice] fmt=Grey1
                    ndarray[H,W,Slice] fmt=GreyStep
                Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
                     ndarray[CountAug,H,W,Slice] fmt=GreyStep
                '''
                atlasesImg, atlasesMask = aug(atlasImg, atlasMask)
            else:
                atlasesImg = np.array([atlasImg])
                atlasesMask = np.array([atlasMask])

            for j in range(atlasesImg.shape[0]):
                imgs = atlasesImg[j]
                masks = atlasesMask[j]
                self.imgDataWrappers = np.append(self.imgDataWrappers, \
                                                 np.array([ImgDataWrapper(imgs, masks, classes, resize=resize, preproc=preproc)]), \
                                                 axis=0)
            print("  Done...")

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlices(slices)

    def SetSlices(self, slices):
        self.slices = slices
        self.len = 0
        for wrapper in self.imgDataWrappers:
            self.len += wrapper.GetImgCount(slices)

    def __DecodeIndex(self, idx):
        idxWrapper = 0
        idxImg = 0
        for i in range(0, self.imgDataWrappers.shape[0]):
            wrapper = self.imgDataWrappers[i]
            wrapperCount = wrapper.GetImgCount(self.slices)
            if idx - wrapperCount >= 0:
                idx -= wrapperCount
            else:
                idxWrapper = i
                idxImg = idx
                break
        return idxWrapper, idxImg

    # Out: ndarray[Slices,(Channels),H,W] dtype=int
    def __getitem__(self, index):  # 返回的是ndarray
        idxWrapper, idxImg = self.__DecodeIndex(index)
        #print("idxWrapper, idxImg: ", idxWrapper, idxImg)
        img0, target0 = self.imgDataWrappers[idxWrapper].Get(idxImg,
                                                             self.slices)  # ndarray[H,W,Slice]  ndarray[H,W,Channel]

        # print("img0",img0.shape)
        # print("target0", target0.shape)
        img = np.transpose(img0, (2, 0, 1))  # ndarray[Slice,H,W]
        target = np.transpose(target0, (2, 0, 1))  # ndarray[Channel,H,W] dtype=int

        # print("target1: ", target1.shape)
        # print("img:\n", type(img), "\n", img.shape)
        # print("target:\n", type(img), "\n", target.shape)
        return img, target

    def __len__(self):
        return self.len

    @staticmethod
    def Split(niisData, niisMask, trainSize):
        idx = np.random.permutation(niisData.shape[0])
        idxSplit = int(len(idx) * trainSize)
        idxLen = len(idx)
        print("Splitting into:")
        print("   Train: ", idx[0:idxSplit])
        print("   Test: ", idx[idxSplit:idxLen])
        niisDataTrain = np.asarray(niisData[idx[0:idxSplit]])
        niisMaskTrain = np.asarray(niisMask[idx[0:idxSplit]])
        niisDataTest = np.asarray(niisData[idx[idxSplit:idxLen]])
        niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
        # print("niisDataTrain\n", niisDataTrain)
        # print("niisMaskTrain\n", niisMaskTrain)
        # print("niisDataTest\n", niisDataTest)
        # print("niisMaskTest\n", niisMaskTest)

        return {'niisDataTrain': niisDataTrain, \
                'niisMaskTrain': niisMaskTrain, \
                'niisDataTest': niisDataTest, \
                'niisMaskTest': niisMaskTest}


# Use Grey1 img (for final storage) ndarray [H,W,Slice]
# Use Onehot mask (for final storage) ndarray [H,W,C,Slice]
# Preproc deal with Grey1 img and GreyStep mask (return the same format)
# TODO: Maybe wrap image into a class
class ImgDataWrapper():
    def __init__(self, imgs, masks, classes, resize=None, preproc=None, imgDataFmt=float, maskDataFmt=int):

        self.imgs = imgs
        self.masks = masks

        if resize is not None:
            # Use pure resizing to preserve data matchiing
            self.imgs = transform.resize(self.imgs, resize + (self.imgs.shape[2],), order=0, clip=True,
                                         preserve_range=True, anti_aliasing=False)
            self.masks = transform.resize(self.masks, resize + (self.masks.shape[2],), order=0, clip=True,
                                          preserve_range=True, anti_aliasing=False)

        if DEBUG:
            testNum = 20
            maskTest255 = ImageProcessor.MapTo255(self.masks[..., testNum], max=classes - 1)
            ImageProcessor.ShowGrayImgHere(maskTest255, "MASK", (10, 10))

        if preproc is not None:
            self.imgs, self.masks = preproc(self.imgs, self.masks, classes)

        self.imgs = self.imgs.astype(dtype=imgDataFmt)
        self.masks = self.masks.astype(dtype=maskDataFmt)

        # One-hot encoding target
        if classes <= 1:
            raise Exception("Class count = 1. Unable to run! ")

        maxClass = np.max(np.unique(self.imgs))
        if maxClass > classes:
            raise Exception("Not enough classes! MaxClassValue: " + str(maxClass))

        self.masks = CommonUtil.PackIntoOneHot(self.masks, classes)

        # TEST
        channel1 = self.masks[..., 1, :]
        # print("channel1",channel1.shape)
        # print("channel1", np.unique(channel1))
        if np.all(channel1 == 0):
            raise Exception("Onehot encoding error! Channel 1 all 0!")
        # ENDTEST

        # print("self.imgs.shape",self.imgs.shape)
        # print("self.masks.shape",self.masks.shape)

        if np.any(np.isnan(self.imgs)):
            raise Exception("NAN Warning!")
        if np.any(np.isnan(self.masks)):
            raise Exception("NAN Warning!")

    # TODO: Form a single dir for each ImgDataWrapper
    def SaveToNPY(self, dir, preFix, suffix=".npy"):
        CommonUtil.Mkdir(dir)
        for i in range(self.imgs.shape[-1]):
            filename = preFix + "_" + str(i) + suffix
            CommonUtil.MkFile(dir, filename)
            np.save(os.path.join(dir, filename), self.imgs[..., i])

    # Out: ndarray [H,W,Slices], ndarray[H,W,Channel]
    def Get(self, idx, slices=1):
        return self.imgs[..., idx:idx + slices], self.masks[..., self.GetCenterIdx(idx, slices)]

    def GetCenterIdx(self, idx, slices=1):
        if slices % 2 == 0:
            raise Exception("Slices count is even")
        return int(idx + int(slices / 2))

    def GetImgCount(self, slices=1):
        return self.imgs.shape[2] - slices + 1

def Test():
    paths = CommonUtil.GetFileFromThisRootDir("../Sources/Data/data_nii/masks/", "nii")
    #path = "../Sources/Data/data_nii/masks/MUTR019_T1anat_j01-labels.nii"
    for path in paths:
        imgs = NiiProcessor.ReadGrey255ImgsFromNii(NiiProcessor.ReadNii(path))#NiiProcessor.ReadOriginalGreyImgsFromNii(NiiProcessor.ReadNii(path))
        for i in range(imgs.shape[2]):
            img = imgs[...,i]
            print(img.shape)
            print(np.unique(img))
            ImageProcessor.ShowGrayImgHere(img,i,(10,10))

if __name__ == '__main__':
    Test()