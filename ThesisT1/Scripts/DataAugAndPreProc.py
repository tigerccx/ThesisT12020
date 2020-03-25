import numpy as np

from Utils import ImageProcessor
from Utils import CV2ImageProcessor

import cv2

from memory_profiler import profile
import gc

'''
  Prepoc:

  Input: imgs Grey1 ndarray [H,W,Slice]
         masks GreyStep ndarray [H,W,Slice]
  Preproc deal with Grey1 img and GreyStep mask (return the same format)
'''
# PreprocDistBG: Distinguish BG from unmarked muscle
# @profile
def PreprocDistBG_TRIAL(imgs, masks, classes):
    thres = 8

    imgsU8 = np.asarray(ImageProcessor.MapTo255(imgs), np.uint8)
    del(imgs)
    gc.collect()

    if classes<3:
        raise Exception("Class count not enough to distinguish bg!")

    # Add a channel to onehot
    masks1 = masks + 1
    del(masks)
    gc.collect()

    for k in range(imgsU8.shape[2]):
        imgsU8[...,k] = cv2.blur(imgsU8[:,:,k], (2,2))
        idxs = np.argwhere(imgsU8[...,k]<=thres)
        for idx in idxs:
            idx = tuple(idx)+(k,)
            val = masks1[idx]
            if val<=1:
                masks1[idx]=0

    # Map all ignored classes to class 1 (unmarked muscle)
    classesOrg = len(np.unique(masks1))
    if classesOrg>3:
        # print("Class count org > class count:")
        # print("Map to 0:")
        for i in range(classesOrg-classes):
            classNum = 2+i
            # print("Class ", classNum, "Map to 1")
            masks1[masks1==classNum] = 1
        # print("Map down:")
        for i in range(classes-2):
            classNum = 2+classesOrg-classes+i
            masks1[masks1==classNum] -= classesOrg-classes
            # print("Class ", classNum, classNum-(classesOrg-classes))

    imgs1 = ImageProcessor.MapTo1(np.asarray(imgsU8, int))
    del(imgsU8)
    gc.collect()

    return imgs1, masks1

# Preproc0: Not distinguish BG from unmarked muscle
# @profile
def Preproc0_TRIAL(imgs, masks, classes):

    imgsU8 = np.asarray(ImageProcessor.MapTo255(imgs), np.uint8)
    del(imgs)
    gc.collect()

    if classes < 2:
        raise Exception("Class count not enough!")

    masks1 = masks

    # Map all ignored classes to class 1 (unmarked muscle)
    classesOrg = len(np.unique(masks1))
    if classesOrg > 2:
        # print("Class count org > class count:")
        # print("Map to 0:")
        for i in range(classesOrg - classes):
            classNum = 1 + i
            # print("Class ", classNum, "Map to 0")
            masks1[masks1 == classNum] = 0
        # print("Map down:")
        for i in range(classes-1):
            classNum = 1+classesOrg-classes+i
            masks1[masks1==classNum] -= classesOrg-classes
            # print("Class ", classNum,  " Map to ",classNum - (classesOrg - classes))

    imgs1 = ImageProcessor.MapTo1(np.asarray(imgsU8, int))
    del (imgsU8)
    gc.collect()

    return imgs1, masks1

def PreprocT4(imgs, masks, classes):
    imgsU8 = np.asarray(ImageProcessor.MapTo255(imgs), np.uint8)
    del imgs
    gc.collect()

    if classes < 2:
        raise Exception("Class count not enough!")

    masks1 = masks

    # Map all ignored classes to class 1 (unmarked muscle)
    masks1[masks1 == 2] = 0

    imgs1 = ImageProcessor.MapTo1(np.asarray(imgsU8, int))
    del imgsU8
    gc.collect()

    return imgs1, masks1

'''
Aug:
    Org:
    dataOrg
    Flip
    dataFlip
    for dataOrg and data Flip:
        Rand x4 : Sheer(-0.1:0.1,-0.1:0.1),Scale(0.95,1.05),Rot(-20,20)
'''

# In: ndarray[H,W,Slice] dtype=uint8
# Out: ndarray[CountAug,H,W,Slice] dtype=uint8
# @profile
def AugCV2(imgsCV2, sheer, scale, rot, isImg):
    interpolation = None
    if isImg:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_NEAREST
    atlasesCV2 = np.empty((1,) + imgsCV2.shape)
    for j in range(imgsCV2.shape[2]):
        atlasesCV2[0][..., j] = CV2ImageProcessor.Rotate(CV2ImageProcessor.ScaleAtCenter(
            CV2ImageProcessor.Sheer(imgsCV2[..., j], sheer, interpolation=interpolation), scale,
            interpolation=interpolation), rot, interpolation=interpolation)
    del(imgsCV2)
    gc.collect()
    return atlasesCV2


# In: ndarray[H,W,Slice] fmt=Grey1
#     ndarray[H,W,Slice] fmt=GreyStep
# Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
#      ndarray[CountAug,H,W,Slice] fmt=GreyStep
# DataAug: function for data augmentation
# @profile
def DataAug_TRIAL(atlasImg, atlasMask, countAug=1):

    #     atlasesImgU8 = AugCV2(imgsU8)
    #     atlasesMaskU8 = AugCV2(masksU8)

    '''
    Aug:
        Org:
        dataOrg
        Flip
        dataFlip
        for dataOrg and data Flip:
            Rand x4 : Sheer(-0.1:0.1,-0.1:0.1),Scale(0.95,1.05),Rot(-20,20)
    '''

    atlasesImgU8 = np.asarray([np.asarray(ImageProcessor.MapTo255(atlasImg), np.uint8)], dtype=np.uint8)
    del (atlasImg)
    gc.collect()
    atlasesMaskU8 = np.asarray([np.asarray(ImageProcessor.MapTo255(atlasMask), np.uint8)], dtype=np.uint8)
    del (atlasMask)
    gc.collect()

    imgsU8Flip = np.empty(atlasesImgU8[0].shape)
    for i in range(atlasesImgU8[0].shape[2]):
        imgsU8Flip[..., i] = CV2ImageProcessor.Flip(atlasesImgU8[0][..., i], 0)
    atlasesImgU8 = np.insert(atlasesImgU8, len(atlasesImgU8), imgsU8Flip, axis=0)

    masksU8Flip = np.empty(atlasesMaskU8[0].shape)
    for i in range(atlasesMaskU8[0].shape[2]):
        masksU8Flip[..., i] = CV2ImageProcessor.Flip(atlasesMaskU8[0][..., i], 0)
    atlasesMaskU8 = np.insert(atlasesMaskU8, len(atlasesMaskU8), masksU8Flip, axis=0)

    for i in range(countAug):
        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-20.0, 20.0)
        atlasesImgU8 = np.concatenate((atlasesImgU8, AugCV2(atlasesImgU8[0], sheer, scale, rot, True)), axis=0)
        atlasesMaskU8 = np.concatenate((atlasesMaskU8, AugCV2(atlasesMaskU8[0], sheer, scale, rot, False)), axis=0)

        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-20.0, 20.0)
        atlasesImgU8 = np.concatenate((atlasesImgU8, AugCV2(imgsU8Flip, sheer, scale, rot, True)), axis=0)
        atlasesMaskU8 = np.concatenate((atlasesMaskU8, AugCV2(masksU8Flip, sheer, scale, rot, False)), axis=0)

    atlasesImg = ImageProcessor.MapTo1(np.asarray(atlasesImgU8, np.int16))
    del(atlasesImgU8)
    gc.collect()
    atlasesMask = ImageProcessor.MapToGreyStep(np.asarray(atlasesMaskU8, np.int16))
    del (atlasesMaskU8)
    gc.collect()

    return atlasesImg, atlasesMask

def DataAug(atlasImg, atlasMask, countAug=1):

    #     atlasesImgU8 = AugCV2(imgsU8)
    #     atlasesMaskU8 = AugCV2(masksU8)

    '''
    Aug:
        Org:
        dataOrg
        Flip
        dataFlip
        for dataOrg and data Flip:
            Rand x4 : Sheer(-0.1:0.1,-0.1:0.1),Scale(0.95,1.05),Rot(-10,10)
    '''

    atlasesImgU8 = np.asarray([np.asarray(ImageProcessor.MapTo255(atlasImg), np.uint8)], dtype=np.uint8)
    del (atlasImg)
    gc.collect()
    atlasesMaskU8 = np.asarray([np.asarray(ImageProcessor.MapTo255(atlasMask), np.uint8)], dtype=np.uint8)
    del (atlasMask)
    gc.collect()

    imgsU8Flip = np.empty(atlasesImgU8[0].shape)
    for i in range(atlasesImgU8[0].shape[2]):
        imgsU8Flip[..., i] = CV2ImageProcessor.Flip(atlasesImgU8[0][..., i], 0)
    atlasesImgU8 = np.insert(atlasesImgU8, len(atlasesImgU8), imgsU8Flip, axis=0)

    masksU8Flip = np.empty(atlasesMaskU8[0].shape)
    for i in range(atlasesMaskU8[0].shape[2]):
        masksU8Flip[..., i] = CV2ImageProcessor.Flip(atlasesMaskU8[0][..., i], 0)
    atlasesMaskU8 = np.insert(atlasesMaskU8, len(atlasesMaskU8), masksU8Flip, axis=0)

    for i in range(countAug):
        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-10.0, 10.0)
        atlasesImgU8 = np.concatenate((atlasesImgU8, AugCV2(atlasesImgU8[0], sheer, scale, rot, True)), axis=0)
        atlasesMaskU8 = np.concatenate((atlasesMaskU8, AugCV2(atlasesMaskU8[0], sheer, scale, rot, False)), axis=0)

        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-10.0, 10.0)
        atlasesImgU8 = np.concatenate((atlasesImgU8, AugCV2(imgsU8Flip, sheer, scale, rot, True)), axis=0)
        atlasesMaskU8 = np.concatenate((atlasesMaskU8, AugCV2(masksU8Flip, sheer, scale, rot, False)), axis=0)

    atlasesImg = ImageProcessor.MapTo1(np.asarray(atlasesImgU8, np.int16))
    del(atlasesImgU8)
    gc.collect()
    atlasesMask = ImageProcessor.MapToGreyStep(np.asarray(atlasesMaskU8, np.int16))
    del (atlasesMaskU8)
    gc.collect()

    return atlasesImg, atlasesMask