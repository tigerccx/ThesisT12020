import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.cm as cm
import copy

import torch

from matplotlib import pylab as plt

class PreProcessor():
    # In: ndarray[H,W] dtype=int   (Seq)
    #     ndarray[H,W] dtype=float (Grey1)
    # Out: ndarray[H,W] dtype=int
    @staticmethod
    def EnhanceMask(mask, img, thres=0.02):
        mask1 = mask+1
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):


class NiiProcessor():

    def __init__(self):
        pass

    @staticmethod
    def ReadNii(path):
        nii = nib.load(path)
        return nii

    @staticmethod
    def ReadImgsFromNii(nii):
        return np.asarray(nii.dataobj)

    @staticmethod
    def ReadGrey255ImgsFromNii(nii):
        imgs = NiiProcessor.ReadImgsFromNii(nii)
        imgsG = ImageProcessor.MapTo255(imgs)
        return imgsG

    @staticmethod
    def ReadGrey1ImgsFromNii(nii):
        imgs = NiiProcessor.ReadImgsFromNii(nii)
        imgsG = ImageProcessor.MapTo1(imgs)
        return imgsG

    @staticmethod
    def ReadOriginalGreyImgsFromNii(nii):
        imgs = NiiProcessor.ReadImgsFromNii(nii)
        return imgs

    @staticmethod
    def ReadImgsFromAllNiiInDir(dir):
        pathFiles = CommonUtil.GetFileFromThisRootDir(dir, "nii")
        imgsAll = None
        for path in pathFiles:
            nii = NiiProcessor.ReadNii(path)
            print(nii.header.get_data_shape())
            imgs = NiiProcessor.ReadImgsFromNii(nii)
            if imgsAll is None:
                imgsAll = np.empty((imgs.shape[0], imgs.shape[1], 0))
            imgsAll = np.append(imgsAll, imgs, 2)
            print(imgsAll.shape)
        print(imgsAll.shape)
        return imgsAll

    @staticmethod
    def ReadAllNiiInDir(dir):
        pathNiis = CommonUtil.GetFileFromThisRootDir(dir, "nii")
        niis = None
        for pathNii in pathNiis:
            nii = NiiProcessor.ReadNii(pathNii)
            if niis is None:
                niis = np.array([nii])
            else:
                niis = np.append(niis, [nii])
        return niis

    @staticmethod
    def SaveImgsIntoNii(imgs, niiRef):
        nii = copy.deepcopy(niiRef)
        nii.dataobj = imgs
        return nii

    @staticmethod
    def SaveNii(dir, niiFileName, nii):
        CommonUtil.__Mkdir(dir)
        nib.save(nii, os.path.join(dir, niiFileName))
        return

    @staticmethod
    def ShowNiiGrey255(nii, size=(10,10)):
        imgsG = NiiProcessor.ReadGreyImgsFromNii(nii)
        plt.figure(figsize=size)
        for i in range(imgsG.shape[2]):
            # img
            img_2d = imgsG[:, :, i]
            img_2d = np.transpose(img_2d, (1, 0))
            plt.imshow(img_2d, cmap='gray')  # 显示图像

            # plt.show()
            plt.pause(0.001)

#Use ndarray [H,W,C]
class ImageProcessor():

    def __init__(self):
        pass

    @staticmethod
    def MapTo255(arr):
        arrMax = np.amax(arr)
        if arrMax==0:
            arrMax = 1
        arrG = np.rint(arr / arrMax * 255)
        return arrG

    @staticmethod
    def MapTo1(arr):
        # print("Check Mem(Start Mapping)")
        # os.system("pause")
        arrMax = np.amax(arr)
        # print("Check Mem(After Max)")
        # os.system("pause")
        arrG = np.asarray(arr) / arrMax
        # print("Check Mem(After div)")
        # os.system("pause")
        #return arrG.astype(float)
        arrG = arrG
        # print("Check Mem(After astype)")
        # os.system("pause")
        return  arrG

    @staticmethod
    def MapToGreyStep(arr):
        #TODO: 0-...->01234...
        arr1 = np.sort(np.unique(arr))
        dic={}
        for i in range(len(arr1)):
            dic[arr1[i]]=i
        arr2 = np.array(arr)
        for key in dic:
            arr2[arr2==key]=dic[key]
        return arr2

    @staticmethod
    def ReadGrayImg(path):
        img = cv2.imread(path, 0)
        return img

    @staticmethod
    def ReadClrImg(path):
        img = cv2.imread(path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        return img

    @staticmethod
    def Rotate90(img, counterClockwise=False):
        img = np.asarray(img)
        img = np.transpose(img, (1, 0, 2))
        if counterClockwise:
            img = np.flip(img, 0)
        else:
            img = np.flip(img, 1)
        return img

    @staticmethod
    def ShowGrayImgHere(img, title, size):
        plt.figure(figsize=size)
        plt.axis("off")
        plt.imshow(img, cmap=cm.gray, vmin=0, vmax=255)
        plt.title(title)
        plt.show()
        return

    @staticmethod
    def ShowClrImgHere(img, title, size):
        plt.figure(figsize=size)
        plt.axis("off")
        plt.imshow(img)
        plt.title(title)
        plt.show()
        return

    @staticmethod
    def SaveGrayImg(dir, imgName, img):
        CommonUtil.Mkdir(dir)
        cv2.imwrite(os.path.join(dir,imgName), img)
        return

    @staticmethod
    def SaveClrImg(dir, imgName, img):
        CommonUtil.Mkdir(dir)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        cv2.imwrite(os.path.join(dir, imgName), img)
        return

class CommonUtil():
    @staticmethod
    def Mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    @staticmethod
    def MkFile(dir,filename):
        CommonUtil.Mkdir(dir)
        if not os.path.isfile(os.path.join(dir,filename)):  # 无文件时创建
            fd = open(filename, mode="w+")
            fd.close()

    @staticmethod
    def GetFileFromThisRootDir(dir, ext=None):
        allfiles = []
        needExtFilter = (ext != None)
        for root, dirs, files in os.walk(dir):
            for filespath in files:
                filepath = os.path.join(root, filespath)
                print(os.path.splitext(filepath))
                extension = os.path.splitext(filepath)[1][1:]
                if needExtFilter and extension in ext:
                    allfiles.append(filepath)
                elif not needExtFilter:
                    allfiles.append(filepath)
        return allfiles

    @staticmethod
    def PackIntoTorchType(a):
        if a is float or a is "float" or a is "float32":
            return torch.float32
        if a is "float16":
            return torch.float16
        if a is "float64":
            return torch.float64

        if a is int or a is "int" or a is "int32":
            return torch.int32
        if a is "int8":
            return torch.int8
        if a is "int16":
            return torch.int16
        if a is "int64":
            return torch.int64

        raise Exception("Type is not included!")

    # In: ndarray[H,W,Slices] dtype=int
    # Out: ndarray[H,W,Channels,Slices] dtype=int
    @staticmethod
    def PackIntoOneHot(imgs, countClasses):
        # self.imgs = np.asarray([[[np.eye(countClasses)[self.imgs[x, y, z]] \
        #                           for z in range(self.imgs.shape[2])] \
        #                          for y in range(self.imgs.shape[1])] \
        #                         for x in range(self.imgs.shape[0])], \
        #                        dtype="bool")
        imgs1 = np.zeros((imgs.shape[0], imgs.shape[1], countClasses, imgs.shape[2]), dtype=int)
        for x in range(imgs.shape[0]):
            for y in range(imgs.shape[1]):
                for z in range(imgs.shape[2]):
                    imgs1[x, y, int(imgs[x, y, z]), z] = True
        return imgs1

    # In: ndarray[H,W,Channels,Slices] dtype=int
    # Out: ndarray[H,W,Slices] dtype=int
    @staticmethod
    def UnpackFromOneHot(imgsOneHot):
        imgs = np.zeros((imgsOneHot.shape[0], imgsOneHot.shape[1], imgsOneHot.shape[3]), dtype=int)
        for x in range(imgsOneHot.shape[0]):
            for y in range(imgsOneHot.shape[1]):
                for z in range(imgsOneHot.shape[3]):
                    oneHot = imgsOneHot[x,y,:,z]
                    idx = np.argwhere(oneHot==1).flatten()
                    imgs[x,y,z] = idx[0]
        return imgs

    # In: ndarray[H,W,Channels,Slices] dtype=int
    # Out: ndarray[H,W,Channels,Slices] dtype=int
    @staticmethod
    def HardMax(imgsOnehot):
        imgsHM = np.array(imgsOnehot, dtype=int)
        for i in range(imgsOnehot.shape[0]):
            for j in range(imgsOnehot.shape[1]):
                for k in range(imgsOnehot.shape[3]):
                    onehot = imgsOnehot[i,j,:,k]
                    omax = onehot.max()
                    onehot = np.where(onehot==omax,1,0)
                    imgsHM[i,j,:,k] = onehot
        return imgsHM

def test():
    pathSrc = "../Sources/Data/data_nii"
    pathTarg = "../Sources/Data/output"
    file = "MUTR019_T1anat_j01.nii"#"MUTR019_DTI_j01.nii"
    fileSeg = "MUTR019_T1anat_j01-labels.nii"

    # niiProc = NiiProcessor("../Sources/Data/data_nii","../Sources/Data/output")
    # img = niiProc.readNii("MUTR019_DTI_j01.nii")
    # print(img)

    # img
    path = os.path.join(pathSrc, file)
    nii = NiiProcessor.ReadNii(path)
    print(nii.header.get_data_shape())
    print(nii.header)
    img = NiiProcessor.ReadGreyImgsFromNii(nii)
    print("img:",img.shape)
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255  # 对所求的像素进行归一化变成0-255范围,这里就是三维数据

    # seg
    imgSeg_path = os.path.join(pathSrc, fileSeg)
    imgSeg = np.asanyarray(nib.load(imgSeg_path).dataobj)
    print("seg:",imgSeg.shape)
    imgSeg_3d_max = np.amax(imgSeg)
    print(imgSeg_3d_max)
    imgSeg = imgSeg / imgSeg_3d_max * 255  # 对所求的像素进行归一化变成0-255范围,这里就是三维数据

    plt.figure(figsize=(10, 10))
    for i in range(img.shape[2]):  # 对切片进行循环


        #seg
        plt.subplot(211)
        imgSeg_2d = imgSeg[:, :, i]  # 取出一张图像
        plt.title(np.unique(imgSeg_2d))
        plt.imshow(imgSeg_2d,cmap='gray') # 显示图像

        #img
        plt.subplot(212)
        img_2d = img[:, :, i]
        img_2d = np.transpose(img_2d, (1, 0))
        plt.imshow(img_2d,cmap='gray')  # 显示图像

        #plt.show()
        plt.pause(0.001)

        # filter out 2d images containing < 10% non-zeros
        # print(np.count_nonzero(img_2d))
        # print("before process:", img_2d.shape)
        # if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:  # 表示一副图像非0个数超过整副图像的10%我们才把该图像保留下来
        #     img_2d = img_2d / 127.5 - 1  # 对最初的0-255图像进行归一化到[-1, 1]范围之内
        #     img_2d = np.transpose(img_2d, (1, 0))  # 这个相当于将图像进行旋转90度
        #     plt.imshow(img_2d)
        #     plt.pause(0.01)

def test1():
    pathSrc = "../Sources/Data/data_nii"
    pathTarg = "../Sources/Data/output"
    print(CommonUtil.GetFileFromThisRootDir(pathSrc, "nii"))

def test2():
    pathSrc = "../Sources/Data/data_nii"
    pathTarg = "../Sources/Data/output"
    file = "MUTR019_T1anat_j01.nii"  # "MUTR019_DTI_j01.nii"
    fileSeg = "MUTR019_T1anat_j01-labels.nii"

    path = os.path.join(pathSrc, fileSeg)
    nii = nib.load(path)
    print(nii.header)
    print(nii.header.get_data_shape())
    print(nii.dataobj.shape)

def test3():
    CommonUtil.Mkdir("Test1")
    CommonUtil.MkFile("","zhizhangpython.txt")


if __name__ == '__main__':
    test3()

