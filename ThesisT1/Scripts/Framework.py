import os

import numpy as np

import torch
from torch import tensor
import torch.nn as tnn
import torch.nn.functional as tfunc
import torch.utils.data as data
import torch.optim as topti

# from torchvision import datasets,transforms
from torchsummary import summary
from matplotlib import pylab as plt

import cv2

import pynvml
import gc

from Utils import NiiProcessor
from Utils import ImageProcessor
from Utils import CommonUtil

from DataStructures import ImgDataSet
from DataStructures import ImgDataWrapper

from LossFunc import MulticlassDiceLoss

#MACRO
DEBUG = False
DEBUG_TEST = False


# Original Network
# file:///D:/Sources/Study/Thesis%20Study/Prof%20E%20Meijering/Deep%20Learning%20for%20Muscle%20Segmentation%20in%20MRI%20and%20DTI%20Images/SEMANTIC%20SEGMENTATION%20OF%20THIGH%20MUSCLE%20USING%202.5D.pdf
class Network(tnn.Module):
    # Network Definition
    def __init__(self, classes):
        super(Network, self).__init__()
        # Conv(↓)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = tnn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = tnn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = tnn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = tnn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = tnn.Conv2d(256, 256, 3, padding=1)

        # BatchNorm
        self.bn1 = tnn.BatchNorm2d(64)
        self.bn2 = tnn.BatchNorm2d(256)
        self.bn3 = tnn.BatchNorm2d(128)
        self.bn4 = tnn.BatchNorm2d(32)

        # Maxpool
        # orch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool1 = tnn.MaxPool2d(2)

        # UpSample
        self.upsamp1 = tnn.Upsample(scale_factor=2)

        # UpConv(↑)
        self.upconv5 = tnn.ConvTranspose2d(256, 256, 3, padding=1)
        self.upconv4 = tnn.ConvTranspose2d(512, 256, 3, padding=1)
        self.upconv3 = tnn.ConvTranspose2d(256, 128, 3, padding=1)
        self.upconv2 = tnn.ConvTranspose2d(192, 128, 3, padding=1)
        self.upconv1 = tnn.ConvTranspose2d(128, 32, 3, padding=1)
        self.upconv0 = tnn.ConvTranspose2d(32, classes, 3, padding=1)

        # AnisoUpconv
        self.anisoUpconv1 = tnn.ConvTranspose2d(32, classes, (1, 11), padding=1)

        # Softmax
        self.softmax2d = tnn.Softmax2d()

    def forward_1(self, input):
        # Layer1(↓)
        x2 = tfunc.leaky_relu(self.conv2(tfunc.leaky_relu(self.conv1(input))))

        # Layer2(↓)
        x5 = tfunc.leaky_relu(self.conv4(tfunc.leaky_relu(self.conv3(self.maxpool1(x2)))))

        # Layer3(→)
        x9 = self.upsamp1(tfunc.leaky_relu(self.upconv5(tfunc.leaky_relu(self.conv5(self.maxpool1(x5))))))

        # Layer4(↑)
        x10 = torch.cat([x5, x9], 1)
        x13 = self.upsamp1(tfunc.leaky_relu(self.upconv3(tfunc.leaky_relu(self.upconv4(x10)))))

        # Layer5(↑)
        x14 = torch.cat([x2, x13], 1)
        x17 = self.upconv0(tfunc.leaky_relu(self.upconv1(tfunc.leaky_relu(self.upconv2(x14)))))

        return x17

    def forward(self, input):
        # Layer1(↓)
        x2 = tfunc.leaky_relu(self.bn1(self.conv2(tfunc.leaky_relu(self.conv1(input)))))

        # Layer2(↓)
        x5 = tfunc.leaky_relu(self.bn2(self.conv4(tfunc.leaky_relu(self.conv3(self.maxpool1(x2))))))

        # Layer3(→)
        x9 = self.upsamp1(tfunc.leaky_relu(self.bn2(self.upconv5(tfunc.leaky_relu(self.conv5(self.maxpool1(x5)))))))

        # Layer4(↑)
        x10 = torch.cat([x5, x9], 1)
        x13 = self.upsamp1(tfunc.leaky_relu(self.bn3(tfunc.leaky_relu(self.upconv3(self.upconv4(x10))))))

        # Layer5(↑)
        x14 = torch.cat([x2, x13], 1)
        x17 = self.upconv0(tfunc.leaky_relu(self.bn4(self.upconv1(tfunc.leaky_relu(self.upconv2(x14))))))

        return x17

# def LossFunc(output, target):
# #     # Loss Function
# #     loss = MulticlassDiceLoss()
# #     return MulticlassDiceLoss()#loss(output, target)

def cross_entropy(input_, target, reduction='elementwise_mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax =tnn.LogSoftmax(dim=1)
    res  =-target * logsoftmax(input_)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res

def LossFunc(output, target):
    # Loss Function
    return cross_entropy(output, target)

# Input: ndarray [IdxInBatch, H, W]
#        ndarray [IdxInBatch, H, W]
# Output: float diceCoef
def diceCoef(input, target):
    N = target.shape[0]
    smooth = 1

    input_flat = np.reshape(input, (N, -1))
    target_flat = np.reshape(target, (N, -1))
    # print("input_flat: ",input_flat)
    # print("input_flat: ", input_flat.shape)
    # print("input_flat: ", np.unique(input_flat))
    # print("target_flat: ",target_flat)
    # print("target_flat: ", target_flat.shape)
    # print("target_flat: ", np.unique(target_flat))

    intersection = input_flat * target_flat
    # print("intersection: ",intersection)
    # print("intersection: ", intersection.shape)
    # print("target_flat: ", np.unique(intersection))

    loss = ((2 * (intersection.sum(1)) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)).sum()/N
    # print("loss: ",loss)
    # loss = 1 - loss.sum() / N

    return loss

# Input: imgs Grey1 ndarray [H,W,Slice]
#        masks GreyStep ndarray [H,W,Slice]
# Preproc deal with Grey1 img and GreyStep mask (return the same format)
def Preproc(imgs, masks):

    thres = 8

    imgs255 = ImageProcessor.MapTo255(imgs)
    imgsU8 = np.asarray(imgs255, np.uint8)
    masks1 = masks + 1

    for k in range(imgsU8.shape[2]):
        imgsU8[...,k] = cv2.blur(imgsU8[:,:,k], (2,2))
        idxs = np.argwhere(imgsU8[...,k]<=thres)
        for idx in idxs:
            idx = tuple(idx)+(k,)
            val = masks1[idx]
            if val<=1:
                masks1[idx]=0
        #masks1[idxs and masks1<=1]=0
        # for i in range(imgsU8.shape[0]):
        #     for j in range(imgsU8.shape[1]):
        #         if imgsU8[i,j,k] <= thres:
        #             if masks1[i,j,k]<=1:
        #                 masks1[i,j,k] = 0

    imgs1 = ImageProcessor.MapTo1(np.asarray(imgsU8, int))
    return imgs1, masks1


'''
Aug:
    Org:
    dataOrg
    Flip
    dataFlip
    for dataOrg and data Flip:
        Rand x4 : Sheer(-0.1:0.1,-0.1:0.1),Scale(0.95,0.105),Rot(-20,20)
'''


# In: ndarray[H,W,Slice] dtype=uint8
# Out: ndarray[CountAug,H,W,Slice] dtype=uint8
def AugCV2(imgsCV2, sheer, scale, rot, isImg):
    interpolation = None
    if isImg:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_NEAREST
    atlasesCV2 = np.empty((0,) + imgsCV2.shape)
    imgsCV2Aug = np.empty(imgsCV2.shape)
    for j in range(imgsCV2.shape[2]):
        imgsCV2Aug[..., j] = CV2ImageProcessor.Rotate(CV2ImageProcessor.ScaleAtCenter(
            CV2ImageProcessor.Sheer(imgsCV2[..., j], sheer, interpolation=interpolation), scale,
            interpolation=interpolation), rot, interpolation=interpolation)

    atlasesCV2 = np.append(atlasesCV2, np.asarray([imgsCV2Aug]), axis=0)
    return atlasesCV2


# In: ndarray[H,W,Slice] fmt=Grey1
#     ndarray[H,W,Slice] fmt=GreyStep
# Out: ndarray[CountAug,H,W,Slice] fmt=Grey1
#      ndarray[CountAug,H,W,Slice] fmt=GreyStep
# DataAug: function for data augmentation
def DataAug(atlasImg, atlasMask, countAug=1):
    imgs255 = ImageProcessor.MapTo255(atlasImg)
    imgsU8 = np.asarray(imgs255, np.uint8)

    masks255 = ImageProcessor.MapTo255(atlasMask)
    masksU8 = np.asarray(masks255, np.uint8)

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

    atlasesImgU8 = np.array([imgsU8], dtype=np.uint8)
    atlasesMaskU8 = np.array([masksU8], dtype=np.uint8)

    imgsU8Flip = np.empty(imgsU8.shape)
    for i in range(imgsU8.shape[2]):
        imgsU8Flip[..., i] = CV2ImageProcessor.Flip(imgsU8[..., i], 0)
    atlasesImgU8 = np.append(atlasesImgU8, np.asarray([imgsU8Flip]), axis=0)

    masksU8Flip = np.empty(masksU8.shape)
    for i in range(masksU8.shape[2]):
        masksU8Flip[..., i] = CV2ImageProcessor.Flip(masksU8[..., i], 0)
    atlasesMaskU8 = np.append(atlasesMaskU8, np.asarray([masksU8Flip]), axis=0)

    for i in range(countAug):
        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-20.0, 20.0)
        atlasesImgU8 = np.append(atlasesImgU8, AugCV2(imgsU8, sheer, scale, rot), axis=0)
        atlasesMaskU8 = np.append(atlasesMaskU8, AugCV2(masksU8, sheer, scale, rot), axis=0)

        sheer = (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        scale = np.random.uniform(0.95, 1.05)
        scale = (scale, scale)
        rot = np.random.uniform(-20.0, 20.0)
        atlasesImgU8 = np.append(atlasesImgU8, AugCV2(imgsU8Flip, sheer, scale, rot, True), axis=0)
        atlasesMaskU8 = np.append(atlasesMaskU8, AugCV2(masksU8Flip, sheer, scale, rot, False), axis=0)

    atlasesImg255 = np.asarray(atlasesImgU8, np.int16)
    atlasesMask255 = np.asarray(atlasesMaskU8, np.int16)

    atlasesImg = ImageProcessor.MapTo1(atlasesImg255)
    atlasesMask = ImageProcessor.MapToGreyStep(atlasesMask255)

    return atlasesImg, atlasesMask



#TODO: Add Temp for processed files
#TODO: Add augmentationn
#TODO: Try transfer learning
#TODO: Use multi-thread for reading annd saving


def RunNN(classes, slices, resize, \
         trainTestSplit, batchSizeTrain, epochs, learningRate, \
         toTrain, toTest, toSaveOutput, \
         pathModel, pathSrc, pathTarg, \
         dataFmt="float32", randSeed=0):
    #
    # Main
    #

    np.random.seed(randSeed)

    TRAIN = toTrain
    TEST = toTest
    SAVE_OUTPUT = toSaveOutput

    pathSrcData = os.path.join(pathSrc, "data")
    pathSrcMask = os.path.join(pathSrc, "masks")

    #   Printing Param
    printLossPerBatch = False

    printLossPerFewBatches = False
    batchPerSummary = 8

    accuracy = None
    dice = None

    if TRAIN or TEST:

        #
        # Choose device
        #
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device("cpu")
        print("Device: ", str(device))

        #
        # Prepare Dataset
        #
        # Load nii

        niisData = NiiProcessor.ReadAllNiiInDir(pathSrcData)
        niisMask = NiiProcessor.ReadAllNiiInDir(pathSrcMask)

        # Split train set and test set
        niisAll = ImgDataSet.Split(niisData, niisMask, trainTestSplit)

        # Create DataLoaders
        # TODO: 使用torchvision?
        # DataLoader return tensor [batch,h,w,c]

        if TRAIN:
            print("Making train set...")
            datasetTrain = ImgDataSet(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices, classes=classes, resize=resize, aug=DataAug, preproc=Preproc)
            print("Making train loader...")
            loaderTrain = data.DataLoader(dataset=datasetTrain, batch_size=batchSizeTrain, shuffle=True)
            print("Done")
        if TEST:
            print("Making test set...")
            datasetTest = ImgDataSet(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices, classes=classes, resize=resize, aug=DataAug, preproc=Preproc)
            print("Making test loader...")
            loaderTest = data.DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)
            print("Done")

        # Prepare Network
        net = Network(classes).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)

        if TRAIN:
            criterion = LossFunc
            optimiser = topti.Adam(net.parameters(), lr=learningRate)  # Minimise the loss using the Adam algorithm.

            for epoch in range(epochs):
                print("Running epoch: ",epoch+1)
                running_loss = 0
                epo_loss = 0
                batchCount = 0

                for i, batch in enumerate(loaderTrain):
                    #print("     Batch: ", i+1)
                    # Get a batch and potentially send it to GPU memory.
                    input = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    inputNP = input.numpy()
                    input = input.to(device)

                    mask = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt)) # Required to be converted from bool to float
                    maskNP = mask.numpy()
                    mask = mask.to(device)

                    if DEBUG:
                        print("INPUT: \n", input.shape)
                        print("  ", input.unique())
                        print("MASK: \n", mask.shape)
                        print("  ", mask.unique())

                    # PyTorch calculates gradients by accumulating contributions to them (useful for
                    # RNNs).  Hence we must manually set them to zero before calculating them.
                    optimiser.zero_grad()

                    # Forward pass through the network.
                    output = net(input)
                    outputNP = output.cpu().detach().numpy()

                    if DEBUG:
                        print("OUTPUT: \n", output.shape)
                        print("  ", output.unique())

                    if DEBUG:
                        for iImg in range(inputNP.shape[0]):
                            img = np.transpose(inputNP[iImg],(1,2,0))
                            ImageProcessor.ShowClrImgHere(img, "E"+str(epoch)+"B"+str(i)+"P"+str(iImg),(10,10))

                            mask_0_5 = maskNP[iImg]
                            mask_0_2 = np.transpose(mask_0_5[0:3], (1,2,0))
                            mask_3_5 = np.transpose(mask_0_5[3:6], (1, 2, 0))
                            ImageProcessor.ShowClrImgHere(mask_0_2, "E" + str(epoch) + "B" + str(i) + "P" + str(iImg)+"_0_2_TARG", (10, 10))
                            ImageProcessor.ShowClrImgHere(mask_3_5, "E" + str(epoch) + "B" + str(i) + "P" + str(iImg) + "_3_5_TARG",(10, 10))

                            out_0_5 = outputNP[iImg]
                            out_0_2 = np.transpose(out_0_5[0:3], (1, 2, 0))
                            out_3_5 = np.transpose(out_0_5[3:6], (1, 2, 0))
                            ImageProcessor.ShowClrImgHere(out_0_2, "E" + str(epoch) + "B" + str(i) + "P" + str(iImg) + "_0_2_OUT",
                                                          (10, 10))
                            ImageProcessor.ShowClrImgHere(out_3_5, "E" + str(epoch) + "B" + str(i) + "P" + str(iImg) + "_3_5_OUT",
                                                          (10, 10))


                    # output = output.flatten(start_dim=2)
                    # mask = mask.flatten(start_dim=2)
                    if DEBUG:
                        print("output: \n", output.shape,"\n",output.dtype)
                        print("mask: \n", mask.shape, "\n",output.dtype)

                    loss = criterion(output, mask)

                    # Calculate gradients.
                    loss.backward()

                    # Minimise the loss according to the gradient.
                    optimiser.step()

                    ls = loss.item()
                    if printLossPerBatch:
                        print("LOSS: ", ls)
                    running_loss += ls
                    epo_loss += ls
                    batchCount+=1

                    if printLossPerFewBatches:
                        if i % batchPerSummary == batchPerSummary-1:
                            print("Summary: Epoch: %2d, Batch: %4d, Loss: %f" % (epoch + 1, i + 1, running_loss / batchPerSummary))
                            running_loss = 0

                print("Epoch Summary: Epoch: %2d, Loss: %f" % (epoch + 1, epo_loss / batchCount))
                print("-" * 30)

            # Save mode
            torch.save(net.state_dict(), pathModel)
            print("Saved model")

        if not TRAIN and TEST:
            net.load_state_dict(torch.load(pathModel))

        if TEST:
            print("Testing...")

            rateCorrect = 0
            dice = np.zeros(classes)
            countBatch = 0
            countImg = 0

            # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
            # computations and reduce memory usage.
            with torch.no_grad():
                for i,batch in enumerate(loaderTest):
                    countBatch += loaderTest.batch_size
                    # Get a batch and potentially send it to GPU memory.
                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    # print("inputs:", inputs.shape)
                    inputsNP = inputs.numpy() #ndarray [IdxInBatch, Channel, H, W]
                    inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0)) #ndarray [H, W, Channel, IdxInBatch]
                    # print(inputsNP1.shape)
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt))
                    #print("masks: ",masks[:, 1, ...].unique())
                    masksNP = masks.numpy() #ndarray [IdxInBatch, Channel, H, W]
                    masksNP1 = np.transpose(masksNP, (2, 3, 1, 0)) #ndarray [H, W, Channel, IdxInBatch]

                    # print(masksNP1.shape)
                    masks = masks.to(device)

                    # Get predictions
                    outputs = net(inputs)  # TODO: Need modify
                    # print("outputs ",outputs.shape)
                    outputsNP = outputs.cpu().numpy() #ndarray [IdxInBatch, Channel, H, W]
                    outputsNP1 = np.transpose(outputsNP, (2, 3, 1, 0)) #ndarray [H, W, Channel, IdxInBatch]
                    # print(outputsNP1.shape)
                    predictsNP1 = CommonUtil.HardMax(outputsNP1) #ndarray [H, W, Channel, IdxInBatch]
                    predictsNP = np.transpose(predictsNP1, (3,2,0,1)) #ndarray [IdxInBatch, Channel, H, W]

                    for i in range(classes):
                        predictNP = predictsNP[:, i, ...]
                        maskNP = masksNP[:, i, ...]
                        dice[i]+=diceCoef(predictNP, maskNP)

                    rateCorrect += np.sum(masksNP1 == predictsNP1).item()/len(masksNP1.flatten())

                    if SAVE_OUTPUT:
                        #Output
                        masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)
                        predictsNP2 = CommonUtil.UnpackFromOneHot(predictsNP1)

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            img = inputsNP1[:,:,int(inputsNP1.shape[2]/2),iImg]
                            img255=ImageProcessor.MapTo255(img)
                            # ImageProcessor.ShowGrayImgHere(img255, "P"+str(iImg),(10,10))
                            ImageProcessor.SaveGrayImg(pathTarg,str(countImg)+"_ORG.jpg",img255)

                            mask = masksNP2[:,:, iImg]
                            mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(mask255, "P" + str(iImg)+"_TARG", (10, 10))
                            ImageProcessor.SaveGrayImg(pathTarg, str(countImg) + "_TARG.jpg", mask255)

                            predict = predictsNP2[:,:, iImg]
                            predict255 = ImageProcessor.MapTo255(predict, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(predict255, "P" + str(iImg)+"_PRED", (10, 10))
                            ImageProcessor.SaveGrayImg(pathTarg, str(countImg) + "_PRED.jpg", predict255)

            print("Done")

            accuracy = rateCorrect / countBatch
            dice = dice / countBatch

    return accuracy, dice





def TestNiiWrapper():
    pathSrc = "../Sources/Data/data_nii"
    pathTarg = "../Sources/Data/output"
    pathSrcData = os.path.join(pathSrc, "data")
    pathSrcMask = os.path.join(pathSrc, "masks")
    pathNiisData = CommonUtil.GetFileFromThisRootDir(pathSrcData, "nii")
    pathNiisMask = CommonUtil.GetFileFromThisRootDir(pathSrcMask, "nii")
    for pathNii in pathNiisData:
        nii = NiiProcessor.ReadNii(pathNii)
        wrap = ImgDataWrapper(nii)
        for i in range(0, wrap.GetImgCount(1)):
            # Test 1 slice
            img = wrap.Get(i)

            imgCC = ImageProcessor.Rotate90(img, True)
            imgCC = np.reshape(imgCC, (img.shape[0], img.shape[1]))
            imgCC = ImageProcessor.MapToGrey255(imgCC)
            ImageProcessor.ShowGrayImgHere(imgCC, str(i), (10, 10))

            imgCW = ImageProcessor.Rotate90(img, False)
            imgCW = np.reshape(imgCW, (img.shape[0], img.shape[1]))
            imgCW = ImageProcessor.MapToGrey255(imgCW)
            ImageProcessor.ShowGrayImgHere(imgCW, str(i), (10, 10))

            # Test 3 slices
            img = wrap.Get(i, 3)

            imgCC = ImageProcessor.Rotate90(img, True)
            imgCC = ImageProcessor.MapTo255(imgCC)
            ImageProcessor.ShowClrImgHere(imgCC, str(i), (10, 10))

            imgCW = ImageProcessor.Rotate90(img, False)
            imgCW = ImageProcessor.MapTo255(imgCW)
            ImageProcessor.ShowClrImgHere(imgCW, str(i), (10, 10))


def TestImgDataSet():
    pathSrc = "../Sources/Data/data_nii"
    pathTarg = "../Sources/Data/output"
    pathSrcData = os.path.join(pathSrc, "data")
    pathSrcMask = os.path.join(pathSrc, "masks")

    niisData = NiiProcessor.ReadAllNiiInDir(pathSrcData)
    niisMask = NiiProcessor.ReadAllNiiInDir(pathSrcMask)

    print("Making ImgNiiData ...")
    imgNiiDataAll = ImgDataSet(niisData, niisMask, slices=3, classes=6, resize=(512,512))

    print("Making Dataloader ...")
    # DataLoader return tensor [batch,h,w,c]
    loaderAll = data.DataLoader(dataset=imgNiiDataAll, batch_size=1, shuffle=True)

    for i, batch in enumerate(loaderAll):
        print("enumerate~")

        # # batch-first
        # img = batch[0].numpy()
        # print(img.shape)
        # img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
        # imgCC = ImageProcessor.Rotate90(img, True)
        # imgCC = np.reshape(imgCC, (img.shape[0], img.shape[1]))
        # imgCC = ImageProcessor.MapTo255(imgCC)
        # ImageProcessor.ShowGrayImgHere(imgCC, str(i), (10, 10))
        #
        # mask = batch[1].numpy()
        # print(mask.shape)
        # print("Mask unique: ", np.unique(mask))
        # mask = mask.reshape((mask.shape[1], mask.shape[2], mask.shape[3]))
        # maskCC = ImageProcessor.Rotate90(mask, True)
        # maskCC = np.reshape(maskCC, (img.shape[0], img.shape[1]))
        # maskCC = ImageProcessor.MapTo255(maskCC)
        # ImageProcessor.ShowGrayImgHere(maskCC, str(i), (10, 10))


def TestNetwork():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network(6).to(device)
    summary(net, (3, 256, 256), 8)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示第一块显卡
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU Info:")
    print("Total (MB): ", meminfo.total / 1024 ** 2)  # 第二块显卡总的显存大小
    print("Used (MB): ", meminfo.used / 1024 ** 2)  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    print("Free (MB): ", meminfo.free / 1024 ** 2)  # 第二块显卡剩余显存大小

    # GPU 256
    # Info:
    # Total(MB): 2048.0
    # Used(MB): 1277.7109375
    # Free(MB): 770.2890625

    # GPU 128
    # Info:
    # Total(MB): 2048.0
    # Used(MB): 667.7109375
    # Free(MB): 1380.2890625

def Main():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toTrain = True
    toTest = True

    toSaveOutput = False#True

    #
    # Param Setting
    #
    #   Running Params
    classes = 7
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"

    pathModel = "./model_temp.pth"  # "./model_D2.pth"#"./model.pth"#"./model_6.pth"
    pathSrc = "../../../Sources/Data/data_nii"
    pathTarg = "../../../Sources/Data/output_temp"  # "../../../Sources/Data/output_D2"#"../../../Sources/Data/output"#"../../../Sources/Data/output_6"

    countRun = 5

    # Test from 1 train sets to 11 train sets
    X = np.arange(1,12,1)
    print(X)
    XRate = X/12
    print(XRate)
    YAcc = np.zeros((len(X),countRun))
    YAccAve = np.zeros(len(X))
    YDiceClasses = np.zeros((len(X),classes,countRun))
    YDiceClassesAve = np.zeros((len(X), classes))

    for i in range(len(XRate)):

        print("\n\n")
        print("="*66)
        print("=" * 66)
        print("Run for Train Size: ",X[i])
        print("\n")

        accAve = 0
        diceAve = np.zeros(classes)

        for count in range(countRun):
            print("\n")
            print("*" * 44)
            print("Run: ", count)
            trainTestSplit = XRate[i]
            accuracy, dice = RunNN(classes, slices, resize, \
                  trainTestSplit, batchSizeTrain, epochs, learningRate, \
                  toTrain, toTest, toSaveOutput, \
                  pathModel, pathSrc, pathTarg, \
                  dataFmt, randSeed)
            print("Classification accuracy: ", accuracy)

            print("Dice Coef:")

            for j in range(classes):
                print("Class ", j, ": ", dice[j])

            YAcc[i,count] = accuracy
            YDiceClasses[i,:,count] = dice

            accAve+=accuracy
            diceAve+=dice

        accAve/=countRun
        diceAve/=countRun

        YAccAve[i] = accAve
        YDiceClassesAve[i,:] = diceAve

    # Plot Acc
    plt.title("Accuracy")
    plt.xlabel("Train Set Size")
    plt.ylabel("Accuracy")
    plt.plot(X, YAccAve, label="Accuracy")

    for count in range(countRun):
        Y = YAcc[:,count]
        plt.scatter(X, Y, marker='x')

    plt.legend()
    plt.show()


    # Plot Dice
    plt.title("Dice")
    plt.subplot(1,2,1)
    plt.xlabel("Train Set Size")
    plt.ylabel("Dice")
    plt.plot(X, YDiceClassesAve[:, 0], label="Dark Background")
    plt.plot(X, YDiceClassesAve[:, 1], label="Other Muscles and Tissues")
    plt.plot(X, YDiceClassesAve[:, 2], label="Class 2")
    plt.plot(X, YDiceClassesAve[:, 3], label="Class 3")
    plt.plot(X, YDiceClassesAve[:, 4], label="Class 4")
    plt.plot(X, YDiceClassesAve[:, 5], label="Class 5")
    plt.plot(X, YDiceClassesAve[:, 6], label="Target Muscle")
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.show()

    for cls in range(classes):
        plt.title("Class "+str(cls))
        plt.xlabel("Train Set Size")
        plt.ylabel("Dice")
        plt.plot(X, YDiceClassesAve[:, cls])
        for count in range(countRun):
            Y = YDiceClasses[:, cls, count]
            plt.scatter(X, Y, marker='x')
        plt.show()


if __name__ == '__main__':
    Main()
    # TestNiiWrapper()
    #TestImgDataSet()
    #TestNetwork()
