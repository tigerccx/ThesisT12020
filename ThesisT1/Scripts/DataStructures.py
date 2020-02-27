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
    def __init__(self, niisData, niisMask, slices=1, classes=2, resize=None, dataFmt=None):
        # TODO: Maybe check data-mask match
        self.imgDataWrappersData = np.empty(0)  # [ImgDataWrapper, ...]
        self.imgDataWrappersMask = np.empty(0)  # [ImgDataWrapper, ...]

        print("Constructing Data...")

        for nii in niisData:
            self.imgDataWrappersData = np.append(self.imgDataWrappersData, np.array([ImgDataWrapper(nii, imgFmt=1, resize=resize, dataFmt=dataFmt)]), axis=0)
            print("  Done...")

        print("Constructing Mask...")
        for nii in niisMask:
            self.imgDataWrappersMask = np.append(self.imgDataWrappersMask, np.array([ImgDataWrapper(nii, imgFmt=-1, classes=classes, resize=resize, dataFmt=dataFmt)]), axis=0)
            print("  Done...")

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlices(slices)
        # print(self.niiWrappersMask)

    # Init from a directory containing .npy files
    # def __init__(self, dir, slices, classes):
    #     self.imgDataWrappersData = np.empty(0)  # [ImgDataWrapper, ...]
    #     self.imgDataWrappersMask = np.empty(0)  # [ImgDataWrapper, ...]
    #     print("Constructing Data...")
    #
    #     dirData = os.path.join(dir,"Data")
    #     dirMask = os.path.join(dir, "Masks")
    #
    #     pathsData = CommonUtil.GetFileFromThisRootDir(dirData, "npy")
    #     pathsMask = CommonUtil.GetFileFromThisRootDir(dirMask, "npy")
    #
    #     for path in pathsData:
    #         self.imgDataWrappersData = np.append(self.imgDataWrappersData, np.array([ImgDataWrapper(path, 1)]), axis=0)
    #         print("  Done...")
    #
    #     print("Constructing Mask...")
    #     for path in pathsMask:
    #         self.imgDataWrappersMask = np.append(self.imgDataWrappersMask, np.array([ImgDataWrapper(nii, -1, classes)]),
    #                                              axis=0)
    #         print("  Done...")
    #
    #     self.slices = 0  # int
    #     self.len = 0  # int
    #     self.SetSlices(slices)

    def SetSlices(self, slices):
        self.slices = slices
        self.len = 0
        for wrapper in self.imgDataWrappersData:
            self.len += wrapper.GetImgCount(slices)

    def __DecodeIndex(self, idx):
        idxWrapper = 0
        idxImg = 0
        for i in range(0, self.imgDataWrappersData.shape[0]):
            wrapper = self.imgDataWrappersData[i]
            wrapperCount = wrapper.GetImgCount(self.slices)
            # print(wrapperCount)
            if idx - wrapperCount >= 0:
                idx -= wrapperCount
            else:
                idxWrapper = i
                idxImg = idx
        return idxWrapper, idxImg

    # Out: ndarray[Slices,(Channels),H,W] dtype=int
    def __getitem__(self, index):  # 返回的是ndarray
        idxWrapper, idxImg = self.__DecodeIndex(index)
        img0 = self.imgDataWrappersData[idxWrapper].Get(idxImg, self.slices)# ndarray[H,W,Slices]
        target0 = self.imgDataWrappersMask[idxWrapper].Get(idxImg, 1) #ndarray[,H,W,(Channels),1]
        target0 = np.reshape(target0, (target0.shape[0], target0.shape[1], target0.shape[2])) #ndarray[,H,W,(Channels)]

        img0, target0 = ImgDataSet.Preproc(img0, target0)

        img = np.transpose(img0, (2, 0, 1)) # ndarray[Slices,H,W]
        target = np.transpose(target0, (2, 0, 1))#ndarray[1,(Channels),H,W] dtype=int
        # print("target1: ", target1.shape)
        # print("img:\n", type(img), "\n", img.shape)
        # print("target:\n", type(img), "\n", target.shape)
        return img, target

    def __len__(self):
        return self.len

    @staticmethod
    def Split(niisData, niisMask, trainSize):
        idx = np.random.permutation(niisData.shape[0])
        idxSplit = np.floor(len(idx) * trainSize).astype(int)
        idxLen = len(idx)
        niisDataTrain = np.asarray(niisData[idx[0:idxSplit]])
        niisMaskTrain = np.asarray(niisMask[idx[0:idxSplit]])
        niisDataTest = np.asarray(niisData[idx[idxSplit:idxLen]])
        niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
        return {'niisDataTrain': niisDataTrain, \
                'niisMaskTrain': niisMaskTrain, \
                'niisDataTest': niisDataTest, \
                'niisMaskTest': niisMaskTest}

    @staticmethod
    def Preproc(imgs, mask):
        thres = 8

        imgs255 = ImageProcessor.MapTo255(imgs)
        imgsU8 = np.asarray(imgs255, np.uint8)
        for i in range(imgsU8.shape[2]):
            imgsU8[:,:,i] = cv2.blur(imgsU8[:,:,i], (2,2))

        mask0 = np.zeros((mask.shape[0],mask.shape[1],1),dtype=int)
        mask1 = np.concatenate((mask0,mask),axis=2)

        for i in range(imgsU8.shape[0]):
            for j in range(imgsU8.shape[1]):
                layers =  imgsU8[i,j,:]
                if len(np.argwhere(layers > thres))==0:
                    if mask1[i,j,1]==1:
                        mask1[i,j,1] = 0
                        mask1[i, j, 0] = 1

        imgs1 = ImageProcessor.MapTo1(np.asarray(imgsU8, int))
        return imgs1, mask1

# Use ndarray [H,W,(C),Slice(C)]
# Use Grey1 img
#TODO: Maybe wrap image into a class
class ImgDataWrapper():
    def __init__(self, nii, imgFmt, classes=1, resize=None, dataFmt=None):
        # print(nii.header)
        # print("Check Header")
        # os.system("pause")
        self.fmt = imgFmt
        self.imgs = None
        if imgFmt == 255:
            self.imgs = NiiProcessor.ReadGrey255ImgsFromNii(nii)
            # print("255")
        elif imgFmt == 1:
            self.imgs = NiiProcessor.ReadGrey1ImgsFromNii(nii)
            # print("1")
        else:
            self.imgs = NiiProcessor.ReadOriginalGreyImgsFromNii(nii)
            # print("orig")

        if resize is not None:
            # Use pure resizing to preserve data matchiing
            self.imgs = transform.resize(self.imgs, resize+(self.imgs.shape[2],), order=0, clip=True,  preserve_range=True, anti_aliasing=False)
            # print("resize")

        if dataFmt is not None and classes<=1:
            self.imgs = self.imgs.astype(dtype=dataFmt)
            # print("astype")

        # One-hot encoding target
        if classes > 1:

            if DEBUG:
                #TEST
                testNum = 20
                print("self.imgs.shape ",self.imgs.shape)
                imgsTest = ImageProcessor.MapTo255(self.imgs)
                imgTest = imgsTest[...,testNum]
                print(np.unique(imgTest))
                ImageProcessor.ShowGrayImgHere(imgTest,imgTest.shape,(10,10))
                imgTest255 = ImageProcessor.MapTo255(imgTest)
                ImageProcessor.ShowGrayImgHere(imgTest255, "255", (10, 10))
                imgTest255 = imgsTest[...,testNum]
                ImageProcessor.ShowGrayImgHere(imgTest255, "255si", (10, 10))
                #ENDTEST

            arrClasses = np.arange(classes)
            countClasses = arrClasses.shape[0]
            max = np.max(np.unique(self.imgs))
            if max>classes:
                raise Exception("Not enough classes! MaxClassValue: "+str(max))

            self.imgs = CommonUtil.PackIntoOneHot(self.imgs, countClasses)

            if DEBUG:
                # TEST
                print("self.imgs: ",self.imgs.shape)
                imgsTest = np.transpose(self.imgs, (3,2,0,1))
                mask_0_2 = np.transpose(imgsTest[testNum, 0:3], (1, 2, 0))
                mask_3_5 = np.transpose(imgsTest[testNum, 3:6], (1, 2, 0))
                ImageProcessor.ShowClrImgHere(mask_0_2, "_0_2_TARG",(10, 10))
                ImageProcessor.ShowClrImgHere(mask_3_5, "_3_5_TARG",(10, 10))
                # ENDTEST


        print(self.imgs.dtype,": ", self.imgs.shape)
        if np.any(np.isnan(self.imgs)):
            raise Exception("NAN Warning!")

    # Init data from .npy
    # def __init__(self, dir, imgFmt, isMultiChannel=False):
    #     self.imgs = None
    #     for path in paths:
    #         if self.imgs is None:
    #             self.imgs = np.array([np.load(path)])
    #         else:
    #             self.imgs = np.append(self.imgs, np.array([np.load(path)]),axis=0)
    #     if isMultiChannel:
    #         self.imgs = np.transpose(self.imgs,(3,0,1,2))
    #     else:
    #         self.imgs = np.transpose(self.imgs, (2,0,1))

    # TODO: Form a single dir for each ImgDataWrapper
    def SaveToNPY(self, dir, preFix, suffix=".npy"):
        CommonUtil.Mkdir(dir)
        for i in range(self.imgs.shape[-1]):
            filename = preFix+"_"+str(i)+suffix
            CommonUtil.MkFile(dir, filename)
            np.save(os.path.join(dir, filename), self.imgs[...,i])

    # Out: ndarray [H,W,Slices]
    def Get(self, idx, slices=1):
        # return self.imgs[idx:idx+slices,:,:]
        return self.imgs[..., idx:idx + slices]

    def GetCenterIdx(self, idx, slices=1):
        if slices % 2 == 0:
            raise Exception("Slices count is even")
        return int(idx + np.floor(slices / 2))

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