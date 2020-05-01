import os

import numpy as np
from skimage import transform
import torch.utils.data as data

from memory_profiler import profile
import sys
import gc

from Utils import NiiProcessor
from Utils import ImageProcessor
from Utils import CommonUtil

#MACRO
DEBUG=False


# Use index-last ndarray for storing data
# Use Grey1 img
class ImgDataSet(data.Dataset):

    def SetSlicesAndDis(self, slices, dis):
        self.slices = slices
        self.len = 0
        self.dis = dis
        for wrapper in self.imgDataWrappers:
            self.len += wrapper.GetImgCount(slices, dis)

    def __DecodeIndex(self, idx):
        idxWrapper = 0
        idxImg = 0
        for i in range(0, self.imgDataWrappers.shape[0]):
            wrapper = self.imgDataWrappers[i]
            wrapperCount = wrapper.GetImgCount(self.slices, self.dis)
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
        # print(idxWrapper, ",", idxImg)
        img0, target0 = self.imgDataWrappers[idxWrapper].Get(idxImg, self.slices, self.dis)  # ndarray[H,W,Slice]  ndarray[H,W,Channel]

        img = np.transpose(img0, (2, 0, 1))  # ndarray[Slice,H,W]
        target = np.transpose(target0, (2, 0, 1))  # ndarray[Channel,H,W] dtype=int

        return img, target

    def __len__(self):
        return self.len

    @staticmethod
    def Split(niisData, niisMask, trainSize, toValidate=False, valiSize=None):
        idx = np.random.permutation(niisData.shape[0])
        idxSplit = int(len(idx) * trainSize)
        idxLen = len(idx)

        if toValidate:
            if valiSize is None:
                raise Exception("Validation size cannot be None")
            idxValSplit = int(idxSplit*valiSize)

            niisDataTrain = np.asarray(niisData[idx[0:idxValSplit]])
            niisMaskTrain = np.asarray(niisMask[idx[0:idxValSplit]])
            niisDataValidate = np.asarray(niisData[idx[idxValSplit:idxSplit]])
            niisMaskValidate = np.asarray(niisMask[idx[idxValSplit:idxSplit]])
            niisDataTest = np.asarray(niisData[idx[idxSplit:idxLen]])
            niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
            print("Splitting into:")
            print("-->Train: ", idx[0:idxValSplit])
            print("-->Validate: ", idx[idxValSplit:idxSplit])
            print("-->Test: ", idx[idxSplit:idxLen])
            return {'niisDataTrain': niisDataTrain, \
                    'niisMaskTrain': niisMaskTrain, \
                    "niisDataValidate": niisDataValidate, \
                    "niisMaskValidate": niisMaskValidate, \
                    'niisDataTest': niisDataTest, \
                    'niisMaskTest': niisMaskTest}

        else:
            niisDataTrain = np.asarray(niisData[idx[0:idxSplit]])
            niisMaskTrain = np.asarray(niisMask[idx[0:idxSplit]])
            niisDataTest = np.asarray(niisData[idx[idxSplit:idxLen]])
            niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
            print("Splitting into:")
            print("-->Train: ", idx[0:idxSplit])
            print("-->Test: ", idx[idxSplit:idxLen])
            return {'niisDataTrain': niisDataTrain, \
                    'niisMaskTrain': niisMaskTrain, \
                    'niisDataTest': niisDataTest, \
                    'niisMaskTest': niisMaskTest}


class ImgDataSetMemory(ImgDataSet):
    # @profile
    def InitFromNiis(self, niisData, niisMask, slices=1, dis=1, classes=2, resize=None, aug=None, preproc=None):
        # TODO: Maybe check data-mask match
        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperMemory)

        print("Constructing Data and Masks...")
        print("-" * 32)
        for i in range(len(niisData)):
            print("  File",i+1,":")
            print("  Loading...")

            niiImg = niisData[i]
            niiMask = niisMask[i]

            atlasesImg = np.asarray([NiiProcessor.ReadGrey1ImgsFromNii(niiImg)])
            atlasesMask = np.asarray([NiiProcessor.ReadGreyStepImgsFromNii(niiMask)])

            gc.collect()

            if aug is not None:
                '''
                aug: function for data augmentation
                In: ndarray[H,W,Slice] fmt=Grey1
                    ndarray[H,W,Slice] fmt=GreyStep
                Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
                     ndarray[CountAug,H,W,Slice] fmt=GreyStep
                '''
                print("  Augmenting...")
                atlasesImg, atlasesMask = aug(atlasesImg[0], atlasesMask[0])

            gc.collect()

            print("  Creating DataWappers...")

            for j in range(atlasesImg.shape[0]):
                self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), \
                                                 ImgDataWrapperMemory(atlasesImg[j], atlasesMask[j], classes,
                                                                      resize=resize, preproc=preproc), \
                                                 axis=0)
            del (atlasesImg)
            del (atlasesMask)
            gc.collect()
            print("  Done...")
            print("  ", "-" * 30)

        # print("=" * 100)
        # for wrapper in self.imgDataWrappers:
        #     print("wrapper:", sys.getsizeof(wrapper.imgs) + sys.getsizeof(wrapper.masks))

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices,dis)

    # @profile
    def InitFromNpys(self, dir, slices=1, dis=1, classes=2):

        self.LoadFromNpys(dir, classes)

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices,dis)

    def SaveToNpys(self, dir):
        countWrappers = len(self.imgDataWrappers)
        digitWrapperCount = len(str(countWrappers - 1))
        dirImg = os.path.join(dir, "imgs")
        dirMask = os.path.join(dir, "masks")
        CommonUtil.Mkdir(dirImg)
        CommonUtil.Mkdir(dirMask)
        for i in range(countWrappers):
            wrapper = self.imgDataWrappers[i]
            imgs = wrapper.imgs
            masks = wrapper.masks
            fn = str(i).zfill(digitWrapperCount) + ".npy"
            pathImg = os.path.join(dirImg, fn)
            pathMask = os.path.join(dirMask, fn)
            np.save(pathImg, imgs)
            np.save(pathMask, masks)

    def LoadFromNpys(self, dir, classes):

        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperMemory)

        npysImgs = CommonUtil.GetFileFromThisRootDir(os.path.join(dir, "imgs"), ".npy")
        npysMasks = CommonUtil.GetFileFromThisRootDir(os.path.join(dir, "masks"), ".npy")
        if len(npysImgs) != len(npysMasks):
            print("WARNING: Count of img inputs is different from count of mask inputs.")
        for i in range(len(npysImgs)):
            # print("Loading",i,":")
            imgs = np.load(npysImgs[i])
            masks = np.load(npysMasks[i])

            if imgs is None:
                raise Exception("Imgs is none. No img has been read.")
            if masks is None:
                raise Exception("Masks is none. No mask has been read.")

            self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), \
                                             ImgDataWrapperMemory(imgs, masks, classes, imgDataFmt=imgs.dtype,
                                                                  maskDataFmt=masks.dtype, isMaskOneHot=True), \
                                             axis=0)


class ImgDataSetMultiTypesMemory(ImgDataSet):
    # Input: arrNiisData ndarray[IdxNii, Type]
    #        niisMask ndarray[IdxNii]
    #        slices, classes int
    #        resize tuple
    #        aug func: Input: ndarray[Type,H,W,Slice] fmt=Grey1
    #                         ndarray[H,W,Slice] fmt=GreyStep
    #                  Output: ndarray[Type,CountAug,H,W,Slice] fmt=Grey1
    #                          ndarray[CountAug,H,W,Slice] fmt=GreyStep)
    #        preproc func: Input: ndarray[H,W,Type,Slices] Grey1
    #                             ndarray [H,W,Slice] GreyStep
    #                             int
    #                      Output: ndarray[H,W,Type,Slices] Grey1
    #                              ndarray [H,W,Slice] GreyStep
    def InitFromNiis(self, arrNiisData, niisMask, slices=1, dis=1, classes=2, resize=None, aug=None, preproc=None):
        # TODO: Maybe check data-mask match
        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperMultiTypesMemory)

        print("Constructing Data and Masks...")
        print("-" * 32)

        for i in range(arrNiisData.shape[0]): # for each sample
            print("  File",i+1,":")
            print("  Loading...")

            arrAtlasesImg = None # ndarray[Type,CountAug,H,W,Slice] fmt=Grey1
            for j in range(arrNiisData.shape[1]): # for each type
                niiImg = arrNiisData[i,j]
                atlasesImg = np.asarray([NiiProcessor.ReadGrey1ImgsFromNii(niiImg)])
                if arrAtlasesImg is None:
                    arrAtlasesImg = np.array([atlasesImg])
                else:
                    arrAtlasesImg = np.insert(arrAtlasesImg, arrAtlasesImg.shape[0], atlasesImg, axis=0)

            niiMask = niisMask[i]
            atlasesMask = np.asarray([NiiProcessor.ReadGreyStepImgsFromNii(niiMask)]) # ndarray[CountAug,H,W,Slice] fmt=GreyStep)

            gc.collect()

            if aug is not None:
                '''
                aug: function for data augmentation
                In: ndarray[H,W,Slice] fmt=Grey1
                    ndarray[H,W,Slice] fmt=GreyStep
                Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
                     ndarray[CountAug,H,W,Slice] fmt=GreyStep
                '''
                print("  Augmenting...")
                arrAtlasesImg, atlasesMask = aug(arrAtlasesImg[:,0,...], atlasesMask[0])

            gc.collect()

            print("  Creating DataWappers...")

            for j in range(arrAtlasesImg.shape[1]):
                self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), \
                                                 ImgDataWrapperMultiTypesMemory(np.transpose(arrAtlasesImg[:,j,...], (1, 2, 0, 3)), atlasesMask[j], classes, resize=resize, preproc=preproc), \
                                                 axis=0)
            del (arrAtlasesImg)
            del (atlasesMask)
            gc.collect()
            print("  Done...")
            print("  ", "-" * 30)

        # print("=" * 100)
        # for wrapper in self.imgDataWrappers:
        #     print("wrapper:", sys.getsizeof(wrapper.imgs) + sys.getsizeof(wrapper.masks))

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices, dis)

    # @profile
    def InitFromNpys(self, dir, slices=1, dis=1, classes=2):

        self.LoadFromNpys(dir, classes)

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices,dis)

    def SaveToNpys(self, dir):
        countWrappers = len(self.imgDataWrappers)
        digitWrapperCount = len(str(countWrappers - 1))
        dirImg = os.path.join(dir, "arrImgs")
        dirMask = os.path.join(dir, "masks")
        CommonUtil.Mkdir(dirImg)
        CommonUtil.Mkdir(dirMask)
        for i in range(countWrappers):
            wrapper = self.imgDataWrappers[i]
            arrImgs = wrapper.arrImgs
            masks = wrapper.masks
            fn = str(i).zfill(digitWrapperCount) + ".npy"
            pathImg = os.path.join(dirImg, fn)
            pathMask = os.path.join(dirMask, fn)
            np.save(pathImg, arrImgs)
            np.save(pathMask, masks)

    def LoadFromNpys(self, dir, classes):

        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperMultiTypesMemory)

        npysImgs = CommonUtil.GetFileFromThisRootDir(os.path.join(dir, "arrImgs"), ".npy")
        npysMasks = CommonUtil.GetFileFromThisRootDir(os.path.join(dir, "masks"), ".npy")
        if len(npysImgs) != len(npysMasks):
            print("WARNING: Count of img inputs is different from count of mask inputs.")
        for i in range(len(npysImgs)):
            # print("Loading",i,":")
            # print("  " + npysImgs[i])
            # print("  " + npysMasks[i])
            imgs = np.load(npysImgs[i])
            masks = np.load(npysMasks[i])
            # print(imgs.shape)
            # print(masks.shape)
            if imgs is None:
                raise Exception("Imgs is none. No img has been read.")
            if masks is None:
                raise Exception("Masks is none. No mask has been read.")

            self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), \
                                             ImgDataWrapperMultiTypesMemory(imgs, masks, classes, imgDataFmt=imgs.dtype,
                                                                  maskDataFmt=masks.dtype, isMaskOneHot=True), \
                                             axis=0)

    @staticmethod
    def Split(arrNiisData, niisMask, trainSize, toValidate=False, valiSize=None):
        idx = np.random.permutation(niisMask.shape[0])
        idxSplit = int(len(idx) * trainSize)
        idxLen = len(idx)

        if toValidate:
            if valiSize is None:
                raise Exception("Validation size cannot be None")
            idxValSplit = int(idxSplit * valiSize)
            print("Splitting into:")
            print("-->Train: ", idx[0:idxValSplit])
            print("-->Validate: ", idx[idxValSplit:idxSplit])
            print("-->Test: ", idx[idxSplit:idxLen])
            arrNiisDataTrain = []
            arrNiisDataValidate = []
            arrNiisDataTest = []
            for i in range(len(arrNiisData)):
                arrNiisDataTrain += [np.asarray(arrNiisData[i][idx[0:idxValSplit]])]
                arrNiisDataValidate += [np.asarray(arrNiisData[i][idx[idxValSplit:idxSplit]])]
                arrNiisDataTest += [np.asarray(arrNiisData[i][idx[idxSplit:idxLen]])]
            arrNiisDataTrain = np.transpose(np.asarray(arrNiisDataTrain),(1,0))
            arrNiisDataValidate = np.transpose(np.asarray(arrNiisDataValidate),(1,0))
            arrNiisDataTest = np.transpose(np.asarray(arrNiisDataTest),(1,0))

            niisMaskTrain = np.asarray(niisMask[idx[0:idxValSplit]])
            niisMaskValidate = np.asarray(niisMask[idx[idxValSplit:idxSplit]])
            niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
            return {'arrNiisDataTrain': arrNiisDataTrain, \
                    'niisMaskTrain': niisMaskTrain, \
                    "arrNiisDataValidate": arrNiisDataValidate, \
                    "niisMaskValidate": niisMaskValidate, \
                    'arrNiisDataTest': arrNiisDataTest, \
                    'niisMaskTest': niisMaskTest}

        else:
            print("Splitting into:")
            print("-->Train: ", idx[0:idxSplit])
            print("-->Test: ", idx[idxSplit:idxLen])
            arrNiisDataTrain = []
            arrNiisDataTest = []
            for i in range(arrNiisData.shape[0]):
                arrNiisDataTrain += [np.asarray(arrNiisData[i][idx[0:idxSplit]])]
                arrNiisDataTest += [np.asarray(arrNiisData[i][idx[idxSplit:idxLen]])]
            arrNiisDataTrain = np.transpose(np.asarray(arrNiisDataTrain),(1,0))
            arrNiisDataTest = np.transpose(np.asarray(arrNiisDataTest),(1,0))

            niisMaskTrain = np.asarray(niisMask[idx[0:idxSplit]])
            niisMaskTest = np.asarray(niisMask[idx[idxSplit:idxLen]])
            return {'arrNiisDataTrain': arrNiisDataTrain, \
                    'niisMaskTrain': niisMaskTrain, \
                    'arrNiisDataTest': arrNiisDataTest, \
                    'niisMaskTest': niisMaskTest}


class ImgDataSetDisk(ImgDataSet):
    def InitFromNiis(self, niisData, niisMask, dirSave, slices=1, dis=1, classes=2, resize=None, aug=None, preproc=None,
                     maxNiiCountDigitAfterAug=4):
        # TODO: Maybe check data-mask match
        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperDisk)

        numNii = 0

        print("Constructing Data and Masks...")
        print("-" * 32)
        for i in range(len(niisData)):
            print("  Loading...")

            niiImg = niisData[i]
            niiMask = niisMask[i]

            atlasesImg = np.asarray([NiiProcessor.ReadGrey1ImgsFromNii(niiImg)])
            atlasesMask = np.asarray([NiiProcessor.ReadGreyStepImgsFromNii(niiMask)])

            gc.collect()

            if aug is not None:
                '''
                aug: function for data augmentation
                In: ndarray[H,W,Slice] fmt=Grey1
                    ndarray[H,W,Slice] fmt=GreyStep
                Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
                     ndarray[CountAug,H,W,Slice] fmt=GreyStep
                '''
                print("  Augmenting...")
                atlasesImg, atlasesMask = aug(atlasesImg[0], atlasesMask[0])

            gc.collect()

            print("  Creating DataWappers...")

            for j in range(atlasesImg.shape[0]):
                if len(str(numNii)) > maxNiiCountDigitAfterAug:
                    raise Exception("WARNING: maxNiiCountDigitAfterAug not enough")

                dirWrapper = os.path.join(dirSave, str(numNii).zfill(maxNiiCountDigitAfterAug))
                imgDataWrapper = ImgDataWrapperDisk()
                imgDataWrapper.InitFromImgsAndMasks(atlasesImg[j], atlasesMask[j], classes, dirWrapper, resize=resize,
                                                    preproc=preproc)
                self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), imgDataWrapper,
                                                 axis=0)
                numNii += 1

            del (atlasesImg)
            del (atlasesMask)
            gc.collect()
            print("  Done...")
            print("  ", "-" * 30)

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices, dis)

    def InitFromDir(self, dirImgDataSet, slices=1, dis=1, classes=2):
        self.imgDataWrappers = np.empty(0, dtype=ImgDataWrapperDisk)

        dirsWrapper = os.listdir(dirImgDataSet)
        for dirWrapper in dirsWrapper:
            dirWrapper1 = os.path.join(dirImgDataSet,dirWrapper)
            imgDataWrapper = ImgDataWrapperDisk()
            imgDataWrapper.InitFromDir(dirWrapper1)
            self.imgDataWrappers = np.insert(self.imgDataWrappers, len(self.imgDataWrappers), imgDataWrapper, axis=0)

        self.slices = 0  # int
        self.len = 0  # int
        self.SetSlicesAndDis(slices, dis)


# Use Grey1 img (for final storage) ndarray [H,W,Slice]
# Use Onehot mask (for final storage) ndarray [H,W,C,Slice]
# Preproc deal with Grey1 img and GreyStep mask (return the same format)
# TODO: Maybe wrap image into a class
class ImgDataWrapper():

    def Get(self, idx, slices=1, dis=1):
        raise NotImplementedError

    def GetImgCount(self, slices=1, dis=1):
        raise NotImplementedError

    def GetCenterIdx(self, idx, slices=1, dis=1):
        if slices % 2 == 0:
            raise Exception("Slices count is even")
        return int(idx + int(slices / 2)*dis)


class ImgDataWrapperMemory(ImgDataWrapper):
    # @profile
    def __init__(self, imgs, masks, classes, resize=None, preproc=None, imgDataFmt=float, maskDataFmt=int,
                 isMaskOneHot=False):

        self.imgs = imgs
        self.masks = masks

        if resize is not None:
            # Use pure resizing to preserve data matchiing
            self.imgs = transform.resize(self.imgs, resize + (self.imgs.shape[2],), order=0, clip=True,
                                         preserve_range=True, anti_aliasing=False)
            self.masks = transform.resize(self.masks, resize + (self.masks.shape[2],), order=0, clip=True,
                                          preserve_range=True, anti_aliasing=False)
            gc.collect()

        if preproc is not None:
            self.imgs, self.masks = preproc(self.imgs, self.masks, classes)

        gc.collect()

        self.imgs = self.imgs.astype(dtype=imgDataFmt)
        self.masks = self.masks.astype(dtype=maskDataFmt)

        # One-hot encoding target
        if classes <= 1:
            raise Exception("Class count = 1. Unable to run! ")

        if not isMaskOneHot:
            maxClass = np.max(np.unique(self.masks))
            if maxClass > classes:
                raise Exception("Not enough classes! MaxClassValue: " + str(maxClass))

            self.masks = CommonUtil.PackIntoOneHot(self.masks, classes)
            gc.collect()

        if np.any(np.isnan(self.imgs)):
            raise Exception("NAN Warning!")
        if np.any(np.isnan(self.masks)):
            raise Exception("NAN Warning!")

    #     def SaveToSingleNPY(self, dir, preFix, suffix=".npy"):
    #         CommonUtil.Mkdir(dir)
    #         for i in range(self.imgs.shape[-1]):
    #             filename = preFix + "_" + str(i) + suffix
    #             CommonUtil.MkFile(dir, filename)
    #             np.save(os.path.join(dir, filename), self.imgs[..., i])

    # Out: ndarray [H,W,Slices], ndarray[H,W,Channel]
    def Get(self, idx, slices=1, dis=1):
        idxs = [idx+i*dis for i in range(slices)]
        return self.imgs[..., idxs], self.masks[..., self.GetCenterIdx(idx, slices, dis)]

    def GetImgCount(self, slices=1, dis=1):
        return self.imgs.shape[2] - int(slices/2)*dis*2


class ImgDataWrapperMultiTypesMemory(ImgDataWrapper):
    # Input: arrImgs ndarray[H, W, Type, Slices] 【Note! Shape changes when stored】
    #        masks ndarray[H, W, Slices]
    #        classes int
    #        resize tuple
    #        preproc func: Input: ndarray[H,W,Type,Slices] Grey1
    #                             ndarray [H,W,Slice] GreyStep
    #                             int
    #                      Output: ndarray[H,W,Type,Slices] Grey1
    #                              ndarray [H,W,Slice] GreyStep
    #        imgDataFmt, maskDataFmt type
    #        isMaskOneHot bool
    def __init__(self, arrImgs, masks, classes, resize=None, preproc=None, imgDataFmt=float, maskDataFmt=int,
                 isMaskOneHot=False):

        self.arrImgs = arrImgs # ndarray[H, W, Type, Slices]
        self.masks = masks

        if resize is not None:
            # Use pure resizing to preserve data matchiing
            self.arrImgs = transform.resize(self.arrImgs, (self.arrImgs.shape[0],)+resize + (self.arrImgs.shape[3],), order=0, clip=True,
                                         preserve_range=True, anti_aliasing=False)
            self.masks = transform.resize(self.masks, resize + (self.masks.shape[2],), order=0, clip=True,
                                          preserve_range=True, anti_aliasing=False)
            gc.collect()

        if preproc is not None:
            self.arrImgs, self.masks = preproc(self.arrImgs, self.masks, classes)

        gc.collect()

        self.arrImgs = self.arrImgs.astype(dtype=imgDataFmt)
        self.masks = self.masks.astype(dtype=maskDataFmt)

        # One-hot encoding target
        if classes <= 1:
            raise Exception("Class count = 1. Unable to run! ")

        if not isMaskOneHot:
            maxClass = np.max(np.unique(self.masks))
            if maxClass > classes:
                raise Exception("Not enough classes! MaxClassValue: " + str(maxClass))

            self.masks = CommonUtil.PackIntoOneHot(self.masks, classes)
            gc.collect()

        if np.any(np.isnan(self.arrImgs)):
            raise Exception("NAN Warning!")
        if np.any(np.isnan(self.masks)):
            raise Exception("NAN Warning!")

    # Out: ndarray[H,W,Type*Slices], ndarray[H,W,Channel]
    #                 [t0s0,t0s1,t0s2,...,t1s0,t1s1,...,...]
    def Get(self, idx, slices=1, dis=1):
        shapeArrImg = self.arrImgs.shape
        idxs = [idx + i * dis for i in range(slices)]
        arrImgs = self.arrImgs[..., idxs].reshape(shapeArrImg[0:2]+(shapeArrImg[2]*slices,))
        masks = self.masks[..., self.GetCenterIdx(idx, slices, dis)]
        return arrImgs, masks

    def GetImgCount(self, slices=1, dis=1):
        return self.masks.shape[-1] - int(slices/2)*dis*2


class ImgDataWrapperDisk(ImgDataWrapper):
    # @profile
    def InitFromImgsAndMasks(self, imgs, masks, classes, dirSave, resize=None, preproc=None, imgDataFmt=float,
                             maskDataFmt=int,
                             isMaskOneHot=False):
        self.pathsImg = None
        self.pathsMasks = None

        if resize is not None:
            # Use pure resizing to preserve data matchiing
            imgs = transform.resize(imgs, resize + (self.imgs.shape[2],), order=0, clip=True,
                                    preserve_range=True, anti_aliasing=False)
            masks = transform.resize(masks, resize + (self.masks.shape[2],), order=0, clip=True,
                                     preserve_range=True, anti_aliasing=False)
            gc.collect()

        if preproc is not None:
            imgs, masks = preproc(imgs, masks, classes)

        gc.collect()

        imgs = imgs.astype(dtype=imgDataFmt)
        masks = masks.astype(dtype=maskDataFmt)

        # One-hot encoding target
        if classes <= 1:
            raise Exception("Class count = 1. Unable to run! ")

        if not isMaskOneHot:
            maxClass = np.max(np.unique(masks))
            if maxClass > classes:
                raise Exception("Not enough classes! MaxClassValue: " + str(maxClass))

            masks = CommonUtil.PackIntoOneHot(masks, classes)
            gc.collect()

        if np.any(np.isnan(imgs)):
            raise Exception("NAN Warning!")
        if np.any(np.isnan(masks)):
            raise Exception("NAN Warning!")

        # Save to npys
        self.__SaveToNPYs(dirSave, imgs, masks)

    def InitFromDir(self, dirLoad):
        self.pathsImg = []
        self.pathsMasks = []

        # Load from npys
        self.__LoadFromNPYs(dirLoad)

    def __LoadFromNPYs(self, dirWrapper):
        dirData = os.path.join(dirWrapper, "data")
        dirMasks = os.path.join(dirWrapper, "masks")
        self.pathsImg = CommonUtil.GetFileFromThisRootDir(dirData, ".npy")
        self.pathsMasks = CommonUtil.GetFileFromThisRootDir(dirMasks, ".npy")

    # Form a single dir for each ImgDataWrapper
    # Form a single file for each slice
    def __SaveToNPYs(self, dirWrapper, imgs, masks):
        self.pathsImg = []
        self.pathsMasks = []

        suffix = ".npy"
        dirData = os.path.join(dirWrapper, "data")
        dirMasks = os.path.join(dirWrapper, "masks")
        CommonUtil.Mkdir(dirData)
        CommonUtil.Mkdir(dirMasks)
        slices = imgs.shape[-1]
        digitCount = len(str(slices))
        for i in range(slices):
            filename = str(i).zfill(digitCount) + suffix
            pathImg = os.path.join(dirData, filename)
            pathMask = os.path.join(dirMasks, filename)
            np.save(pathImg, imgs[..., i])
            self.pathsImg += [pathImg]
            np.save(pathMask, masks[..., i])
            self.pathsMasks += [pathMask]

    # Out: ndarray [H,W,Slices], ndarray[H,W,Channel]
    def Get(self, idx, slices=1, dis=1):
        imgsGet = None
        idxs = [idx + i * dis for i in range(slices)]

        for i in idxs:
            img = np.load(self.pathsImg[i])
            if imgsGet is None:
                imgsGet = np.asarray([img])
            else:
                imgsGet = np.insert(imgsGet, len(imgsGet), img, axis=0)

        maskGet = np.load(self.pathsMasks[self.GetCenterIdx(idx, slices, dis)])

        imgsGet = np.transpose(imgsGet, (1, 2, 0))
        return imgsGet, maskGet

    def GetImgCount(self, slices=1, dis=1):
        return len(self.pathsImg) - int(slices/2)*dis*2

if __name__ == '__main__':
    pass