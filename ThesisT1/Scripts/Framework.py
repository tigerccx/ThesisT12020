import os
import time
from threading import Thread

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

import pynvml
import gc

from Utils import NiiProcessor
from Utils import ImageProcessor
from Utils import CommonUtil
from Utils import CV2ImageProcessor

from DataStructures import ImgDataSet
from DataStructures import ImgDataSetDisk
from DataStructures import ImgDataSetMemory
from DataStructures import ImgDataSetMultiTypesMemory
from DataStructures import ImgDataWrapperDisk
from DataStructures import ImgDataWrapperMemory

from DataAugAndPreProc import PreprocDistBG_TRIAL
from DataAugAndPreProc import Preproc0_TRIAL
from DataAugAndPreProc import PreprocT4
from DataAugAndPreProc import PreprocMultiNiis
from DataAugAndPreProc import DataAug
from DataAugAndPreProc import DataAugMultiNiis

from Network import UNet_0
from Network import UNet_1

from LossFunc import MulticlassDiceLoss

#MACRO
DEBUG = False
DEBUG_TEST = False
DEBUG_SHOW_INPUT = False
OUTPUT_PROB = False # To output the prediction as probability
OUTPUT_NII = True # To output the prediction as .nii files

# Abort training
_abort = None
def anyKeyToAbort():
    global _abort
    _abort = input()

# Init network weight
def weight_init(m):
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, tnn.Conv2d) or isinstance(m, tnn.ConvTranspose2d):
        tnn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
     # 是否为批归一化层
    elif isinstance(m, tnn.BatchNorm2d):
        tnn.init.constant_(m.weight, 1)
        tnn.init.constant_(m.bias, 0)

# CE
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

#
# Loss Func wrapping
#
def LossFunc():
    # Loss Function
    return cross_entropy
    # return DiceLossF
    # return CE_Dice_LossF

def DiceLossF(output, target):
    muldice = MulticlassDiceLoss()
    return muldice(tnn.Softmax(dim=1)(output), target)

def CE_Dice_LossF(output, target,scale=1e-3):
    muldice = MulticlassDiceLoss()
    return scale*cross_entropy(output,target)+muldice(tnn.Softmax(dim=1)(output), target)


# Input: Tensor [IdxInBatch, H, W]
#        Tensor [IdxInBatch, H, W]
# Output: float diceCoef
def diceCoefTorch(input, target):
    smooth = 1
    N = target.shape[0]
    input_flat = input.view((N, -1))
    target_flat = target.view((N, -1))

    intersection = input_flat * target_flat

    # print("input_flat",input_flat.shape)
    # print("target_flat", target_flat.shape)
    # print("intersection", intersection.shape)
    # print(torch.sum(intersection,1))
    # print(2 * (torch.sum(intersection,1)) + smooth)
    loss = torch.sum((2 * (torch.sum(intersection,1)) + smooth) / (torch.sum(input_flat,1) + torch.sum(target_flat,1) + smooth))
    # print(loss)

    return loss.item()

def diceCoefAveTorch(input, target):
    N = target.shape[0]
    return diceCoefTorch(input,target)/ N


# Some util funcs
def MakeImgsToSave(countIW, datasetTest):
    iwCur = datasetTest.imgDataWrappers[countIW]
    lenIW = iwCur.GetImgCount(slices=datasetTest.slices, dis=datasetTest.dis)
    idxBeg = iwCur.GetCenterIdx(0, slices=datasetTest.slices, dis=datasetTest.dis)
    idxEnd = iwCur.GetCenterIdx(lenIW-1, slices=datasetTest.slices, dis=datasetTest.dis)+1
    imgNiiToSave = np.empty(iwCur.imgs.shape, dtype=int)
    maskNiiToSave = np.empty(iwCur.masks.shape[0:2]+(iwCur.masks.shape[3],), dtype=int)
    imgNiiToSave[..., 0:idxBeg] = 0
    imgNiiToSave[..., idxEnd:] = 0
    maskNiiToSave[..., 0:idxBeg] = 0
    maskNiiToSave[..., idxEnd:] = 0
    return imgNiiToSave, maskNiiToSave, idxBeg, idxEnd, lenIW

def SaveBoxPlot(data, pathSave):
    plt.boxplot(data)
    plt.savefig(pathSave)
    plt.cla()

# Main Framework
# Input:
# classes int: count of target classes
# slices int: count of input slices
# dis int: slice-to-slice distance
# resize (int,int): resize shape
# aug func: Input: ndarray[H,W,Slice] fmt=Grey1
#                  ndarray[H,W,Slice] fmt=GreyStep
#           Output: ndarray[CountAug,H,W,Slice] fmt=Grey1
#                   ndarray[CountAug,H,W,Slice] fmt=GreyStep)
#         : Augmentation func
# preproc func: Input: ndarray[H,W,Slices] Grey1
#                      ndarray [H,W,Slice] GreyStep
#                      int
#               Output: ndarray[H,W,Type,Slices] Grey1
#                       ndarray [H,W,Slice] GreyStep
#             : Preproc func
# trainTestSplit float: dataset split rate (train/all)
# batchSizeTrain int: batch size
# epochs int: count of epochs
# learningRate float: learning rate
# toSaveData bool: to save augmentation result on disk
# toLoadData bool: to load augmentation result on disk
# toTrain bool: to train the model
# toSaveRunnningLoss bool: to save running status line plot
# toTest bool: to test the model
# toSaveOutput bool: to save output
# toSaveResultAnalysis bool: to save the result info for analysis (acc, dice of each input and box plots for now)
# dirSaveData str: the directory to save augmentation result on disk
# pathModel str: the path to save the model to
# dirSrc str: the directory to read input
#             Note: dirSrc should contain 2 folders: dirSrc/data and dirSrc/masks to contain input images and input masks
# dirTarg str: the directory to save testing output
# dirRAPlot str: the directory to save the result info for analysis
# pathRunningLossPlot str: the path to save the running loss plot
# pathRunningAccPlot str: the path to save the running acc plot
# pathRunningDicePlot str: the path to save the running dice plot
# toUseDisk bool: to enable memory saving mode (!!!extremely slow!!!)
# dataFmt str: the data format to use for the network (!!!deprecated!!!)
# randSeed int: random seed for numpy
# toPrintTime bool: to print the time each section uses
# toValidate bool: to enable training validation
# trainValidationSplit: dataset split rate (validation/(all-testing))
def RunNN(classes, slices, dis, resize,
          aug, preproc,
          trainTestSplit, batchSizeTrain, epochs, learningRate,
          toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput, toSaveResultAnalysis,
          dirSaveData, pathModel, dirSrc, dirTarg, dirRAPlot, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
          toUseDisk=False, dataFmt="float32", randSeed=0, toPrintTime=True,
          toValidate=True, trainValidationSplit=0.85):
    #
    # Main
    #

    np.random.seed(randSeed)

    SAVE_DATA = toSaveData
    LOAD_DATA = toLoadData
    TRAIN = toTrain
    VALIDATE = toValidate
    TEST = toTest
    SAVE_OUTPUT = toSaveOutput
    SAVE_LOSS = toSaveRunnningLoss
    SAVE_RA = toSaveResultAnalysis
    USE_DISK = toUseDisk
    PRINT_TIME = toPrintTime

    dirSrcData = os.path.join(dirSrc, "data")
    dirSrcMask = os.path.join(dirSrc, "masks")
    dirTrain = "train"
    dirVali = "vali"
    dirTest = "test"
    dirSaveDataTrain = os.path.join(dirSaveData, dirTrain)
    dirSaveDataVali = os.path.join(dirSaveData, dirVali)
    dirSaveDataTest = os.path.join(dirSaveData, dirTest)

    #   Printing Param
    printLossPerBatch = False

    accuracy = None
    dice = None

    if PRINT_TIME:
        startProc = time.time()

    if ((TRAIN or TEST) and not LOAD_DATA) or SAVE_DATA or (TEST and SAVE_OUTPUT and OUTPUT_NII):
        #
        # Prepare Dataset
        #
        # Load nii

        niisData = NiiProcessor.ReadAllNiiInDir(dirSrcData)
        niisMask = NiiProcessor.ReadAllNiiInDir(dirSrcMask)

        print()
        print("Niis to read: ")
        print(niisData)
        print(niisMask)
        # Split train set and test set
        if VALIDATE:
            niisAll = ImgDataSet.Split(niisData, niisMask, trainTestSplit, toValidate=True, valiSize=trainValidationSplit)
        else:
            niisAll = ImgDataSet.Split(niisData, niisMask, trainTestSplit)

    if SAVE_DATA:

        # Create and save training data

        if PRINT_TIME:
            startMkTrain = time.time()
        print("Making training set...")

        if not USE_DISK:
            datasetTrain = ImgDataSetMemory()
            datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices, dis=dis,
                                      classes=classes, resize=resize, aug=aug, preproc=preproc)
        else:
            datasetTrain = ImgDataSetDisk()
            datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], dirSaveDataTrain,
                                      slices=slices, dis=dis,
                                      classes=classes, resize=resize, aug=aug, preproc=preproc)

        print("Done")
        if PRINT_TIME:
            endMkTrain = time.time()
            print("  Making training set took:", CommonUtil.DecodeSecondToFormatedString(endMkTrain - startMkTrain))

        if not USE_DISK:
            if PRINT_TIME:
                startSaveTrainData = time.time()
            print("Saving train data...")
            CommonUtil.Mkdir(dirSaveDataTrain)
            datasetTrain.SaveToNpys(dirSaveDataTrain)
            print("Done")
            if PRINT_TIME:
                endSaveTrainData = time.time()
                print("  Saving train data took:",
                      CommonUtil.DecodeSecondToFormatedString(endSaveTrainData - startSaveTrainData))

            del datasetTrain
            gc.collect()

        if VALIDATE:

            # Create and save validation data

            if PRINT_TIME:
                startMkVali = time.time()
            print("Making validation set...")

            if not USE_DISK:
                datasetVali = ImgDataSetMemory()
                datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], slices=slices, dis=dis,
                                          classes=classes, resize=resize, aug=aug, preproc=preproc)
            else:
                datasetVali = ImgDataSetDisk()
                datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], dirSaveDataVali,
                                          slices=slices, dis=dis, classes=classes, resize=resize, aug=aug, preproc=preproc)

            print("Done")
            if PRINT_TIME:
                endMkVali = time.time()
                print("  Making validation set took:", CommonUtil.DecodeSecondToFormatedString(endMkVali - startMkVali))

            if not USE_DISK:
                if PRINT_TIME:
                    startSaveValiData = time.time()
                print("Saving validation data...")
                CommonUtil.Mkdir(dirSaveDataVali)
                datasetVali.SaveToNpys(dirSaveDataVali)
                print("Done")
                if PRINT_TIME:
                    endSaveValiData = time.time()
                    print("  Saving validation data took:",
                          CommonUtil.DecodeSecondToFormatedString(endSaveValiData - startSaveValiData))

                del datasetVali
                gc.collect()

        # Create and save test data

        if PRINT_TIME:
            startMkTest = time.time()
        print("Making test set...")
        if not USE_DISK:
            datasetTest = ImgDataSetMemory()
            datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices, dis=dis,
                                     classes=classes, resize=resize, aug=aug, preproc=preproc)
        else:
            datasetTest = ImgDataSetDisk()
            datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], dirSaveDataTest,
                                     slices=slices, dis=dis,
                                     classes=classes, resize=resize, aug=aug, preproc=preproc)
        print("Done")
        if PRINT_TIME:
            endMkTest = time.time()
            print("  Making test set took:", CommonUtil.DecodeSecondToFormatedString(endMkTest - startMkTest))

        if not USE_DISK:
            if PRINT_TIME:
                startSaveTestData = time.time()
            print("Saving test data...")
            CommonUtil.Mkdir(dirSaveDataTest)
            datasetTest.SaveToNpys(dirSaveDataTest)
            print("Done")
            if PRINT_TIME:
                endSaveTestData = time.time()
                print("  Saving test data took:",
                      CommonUtil.DecodeSecondToFormatedString(endSaveTestData - startSaveTestData))

            del datasetTest
            gc.collect()

    if TRAIN or TEST:

        #
        # Choose device
        #
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device("cpu")
        print()
        print("Device: ", str(device))

        # Create DataLoaders
        # TODO: 使用torchvision?
        # DataLoader return tensor [batch,h,w,c]

        # Prepare Network
        net = UNet_1(16, slices, classes, depth=5, inputHW=None, dropoutRate=0.5).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)#UNet_0(slices,classes).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)

        if TRAIN:

            # Load training data / Mk training data
            if LOAD_DATA or SAVE_DATA:
                if PRINT_TIME:
                    startLoadTrain = time.time()
                print()
                print("Loading training set...")
                if not USE_DISK:
                    datasetTrain = ImgDataSetMemory()
                    datasetTrain.InitFromNpys(dirSaveDataTrain, slices=slices, dis=dis, classes=classes)
                else:
                    datasetTrain = ImgDataSetDisk()
                    datasetTrain.InitFromDir(dirSaveDataTrain, slices=slices, dis=dis, classes=classes)
                    print(datasetTrain.imgDataWrappers.__len__())
                print("Done")
                if PRINT_TIME:
                    endLoadTrain = time.time()
                    print("  Loading training set took:",
                          CommonUtil.DecodeSecondToFormatedString(endLoadTrain - startLoadTrain))

                if VALIDATE:
                    if PRINT_TIME:
                        startLoadVali = time.time()
                    print()
                    print("Loading validation set...")
                    if not USE_DISK:
                        datasetVali = ImgDataSetMemory()
                        datasetVali.InitFromNpys(dirSaveDataVali, slices=slices, dis=dis, classes=classes)
                    else:
                        datasetVali = ImgDataSetDisk()
                        datasetVali.InitFromDir(dirSaveDataVali, slices=slices, dis=dis, classes=classes)
                        print(datasetVali.imgDataWrappers.__len__())
                    print("Done")
                    if PRINT_TIME:
                        endLoadVali = time.time()
                        print("  Loading validation set took:",
                              CommonUtil.DecodeSecondToFormatedString(endLoadVali - startLoadVali))
            else:
                if PRINT_TIME:
                    startMkTrain = time.time()
                print("Making training set...")

                if not USE_DISK:
                    datasetTrain = ImgDataSetMemory()
                    datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices, dis=dis,
                                              classes=classes, resize=resize, aug=aug, preproc=preproc)
                else:
                    datasetTrain = ImgDataSetDisk()
                    datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], dirSaveDataTrain,
                                              slices=slices, dis=dis, classes=classes, resize=resize, aug=aug, preproc=preproc)

                print("Done")
                if PRINT_TIME:
                    endMkTrain = time.time()
                    print("  Making training set took:",
                          CommonUtil.DecodeSecondToFormatedString(endMkTrain - startMkTrain))

                if VALIDATE:

                    # Create and save validation data

                    if PRINT_TIME:
                        startMkVali = time.time()
                    print("Making validation set...")

                    if not USE_DISK:
                        datasetVali = ImgDataSetMemory()
                        datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"],
                                                 slices=slices, dis=dis, classes=classes, resize=resize, aug=aug, preproc=preproc)
                    else:
                        datasetVali = ImgDataSetDisk()
                        datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], dirSaveDataVali,
                                                 slices=slices, dis=dis, classes=classes, resize=resize, aug=aug, preproc=preproc)

                    print("Done")
                    if PRINT_TIME:
                        endMkVali = time.time()
                        print("  Making validation set took:",
                              CommonUtil.DecodeSecondToFormatedString(endMkVali - startMkVali))

            # Make loader
            print()
            print("Making train loader...")
            loaderTrain = data.DataLoader(dataset=datasetTrain, batch_size=batchSizeTrain, shuffle=True)
            print("Done")
            if VALIDATE:
                print()
                print("Making validation loader...")
                loaderVali = data.DataLoader(dataset=datasetVali, batch_size=batchSizeTrain, shuffle=True)
                print("Done")

            # Train

            if PRINT_TIME:
                startTrain = time.time()
            print()
            print("Training...")
            net.apply(weight_init)
            criterion = LossFunc()
            optimiser = topti.Adam(net.parameters(), lr=learningRate)  # Minimise the loss using the Adam algorithm.

            if SAVE_LOSS:
                runningLoss = np.zeros(epochs, dtype=np.float32)
                runningAcc = np.zeros(epochs, dtype=np.float32)
                runningDice = np.zeros((epochs, classes), dtype=np.float32)
                if VALIDATE:
                    runningLossVali = np.zeros(epochs, dtype=np.float32)
                    runningAccVali = np.zeros(epochs, dtype=np.float32)
                    runningDiceVali = np.zeros((epochs, classes), dtype=np.float32)

            global _abort
            _abort = None
            thd = Thread(target=anyKeyToAbort)
            thd.daemon = True
            thd.start()
            print("NOTE: Input anything during training to abort before the next epoch.")

            params = {}

            weightAveLast = {}
            gradAveLast = {}

            # Run epochs
            for epoch in range(epochs):

                # for name, param in net.named_parameters():
                #     if epoch==0:
                #         params[name]=param
                #     else:
                #         print(params[name]-param)
                #         params[name] = param
                #     # print(name, param)

                if PRINT_TIME:
                    startEpoch = time.time()
                print("Running epoch: ",epoch+1)

                # Run validation batch
                if VALIDATE:
                    epoLossVali = 0
                    sampleCountVali = 0
                    batchCountVali = 0
                    accVali = 0
                    diceVali = np.zeros(classes)
                    with torch.no_grad():
                        for i, batch in enumerate(loaderVali):

                            # print("     Validation Batch: ", i+1)
                            inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                            inputs = inputs.to(device)

                            masks = batch[1].type(
                                CommonUtil.PackIntoTorchType(dataFmt))  # Required to be converted from bool to float
                            masks = masks.to(device)

                            net.eval()
                            outputs = net(inputs)

                            # Loss
                            loss = criterion(outputs, masks)
                            ls = loss.item()
                            if printLossPerBatch:
                                print("VALI LOSS: ", ls)
                            epoLossVali += ls

                            # Acc
                            predicts = torch.zeros_like(outputs)
                            predicts = predicts.scatter(1, torch.max(outputs, 1, keepdim=True).indices, 1)
                            accVali += torch.sum(masks == predicts).item() / masks.numel()

                            # Dice
                            for j in range(classes):
                                predictClass = predicts[:, j, ...]
                                maskClass = masks[:, j, ...]
                                diceVali[j] += diceCoefAveTorch(predictClass, maskClass)

                            sampleCountVali += inputs.shape[0]
                            batchCountVali += 1

                            gc.collect()

                epoLoss = 0
                acc = 0
                dice = np.zeros(classes)
                sampleCount = 0
                batchCount = 0

                if DEBUG_SHOW_INPUT:
                    countImg = 0
                    dirINPUTSAVE = "E:/Project/Python/ThesisT1/Sources/DEBUG_INPUT"

                # Train batch
                for i, batch in enumerate(loaderTrain):
                    #print("     Batch: ", i+1)
                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt)) # Required to be converted from bool to float
                    masks = masks.to(device)

                    if DEBUG_SHOW_INPUT:
                        CommonUtil.Mkdir(dirTarg)

                        # Output
                        inputsNP = inputs.detach().cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]

                        masksNP = masks.detach().cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        masksNP1 = np.transpose(masksNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)

                        if OUTPUT_PROB:
                            outputsNP = tnn.Softmax(dim=1)(outputs)[:,1,...].cpu().numpy()
                            outputsNP1 = np.transpose(outputsNP, (1,2,0))  # ndarray [H, W, Channel, IdxInBatch]

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            img = inputsNP1[:,:,int(inputsNP1.shape[2]/2),iImg]
                            img255=ImageProcessor.MapTo255(img)
                            # ImageProcessor.ShowGrayImgHere(img255, "P"+str(iImg),(10,10))
                            ImageProcessor.SaveGrayImg(dirINPUTSAVE, str(countImg) + "_ORG.jpg", img255)

                            mask = masksNP2[:,:, iImg]
                            mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(mask255, "P" + str(iImg)+"_TARG", (10, 10))
                            ImageProcessor.SaveGrayImg(dirINPUTSAVE, str(countImg) + "_TARG.jpg", mask255)

                    optimiser.zero_grad()

                    # print("inputs.shape",inputs.shape)
                    net.train()
                    outputs = net(inputs)
                    # print("outputs.shape", outputs.shape)
                    # print("masks.shape", outputs.shape)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimiser.step()

                    # Loss
                    ls = loss.item()
                    if printLossPerBatch:
                        print("LOSS: ", ls)
                    epoLoss += ls

                    # Acc
                    predicts = torch.zeros_like(outputs)
                    predicts = predicts.scatter(1,torch.max(outputs, 1, keepdim=True).indices,1)
                    acc += torch.sum(masks == predicts).item() / masks.numel()

                    # Dice
                    for j in range(classes):
                        predictClass = predicts[:, j, ...]
                        maskClass = masks[:, j, ...]
                        dice[j] += diceCoefAveTorch(predictClass, maskClass)

                    sampleCount+= inputs.shape[0]
                    batchCount += 1

                    # print("    ", "+" * 100)
                    # for name, params in net.named_parameters():
                    #     print(type(params))
                    #     weightAve = torch.mean(params.data)
                    #     gradAve = torch.mean(params.grad)
                    #
                    #     print('     -->name:', name, '-->grad_requirs:', params.requires_grad, '--weight',
                    #           weightAve, ' -->grad_value:', gradAve)
                    #
                    #     if name not in weightAveLast:
                    #         weightAveLast[name] = weightAve
                    #     else:
                    #         print("    -->name:", name, "-->weight diff:",weightAve-weightAveLast[name])
                    #         weightAveLast[name] = weightAve
                    #     gradAveLast[name] = gradAve
                    #     if gradAve==0:
                    #         print("            WARNING: zero_grad!")

                    gc.collect()

                epoLoss = epoLoss / batchCount
                acc = acc / batchCount
                dice = dice / batchCount
                if VALIDATE:
                    epoLossVali = epoLossVali / batchCountVali
                    accVali = accVali / batchCountVali
                    diceVali = diceVali / batchCountVali
                    print("Epoch Summary:")
                    print("  Loss: %f, Loss Vali: %f" % (epoLoss, epoLossVali))
                    print("  Acc: %f, Acc Vali: %f" % (acc, accVali))
                    print("  Dice | Dice Vali:")
                    for cls in range(classes):
                        print("    Class %d: %f | %f" % (cls,dice[cls],diceVali[cls]))
                else:
                    print("Epoch Summary:")
                    print("  Loss: %f" % (epoLoss))
                    print("  Acc: %f" % (acc))
                    print("  Dice | Dice Vali:")
                    for cls in range(classes):
                        print("    Class %d: %f" % (cls, dice[cls]))
                if PRINT_TIME:
                    endEpoch = time.time()
                    print("  Epoch took:", CommonUtil.DecodeSecondToFormatedString(endEpoch - startEpoch))
                print("-" * 30)

                if SAVE_LOSS:
                    runningLoss[epoch] = epoLoss
                    runningAcc[epoch] = acc
                    runningDice[epoch,:]=dice
                    if VALIDATE:
                        runningLossVali[epoch] = epoLossVali
                        runningAccVali[epoch] = accVali
                        runningDiceVali[epoch, :] = diceVali
                if _abort is not None:
                    # Abort
                    epochs = epoch+1
                    runningLoss = runningLoss[:epochs]
                    runningAcc = runningAcc[:epochs]
                    runningDice = runningDice[:epochs]
                    if VALIDATE:
                        runningLossVali = runningLossVali[:epochs]
                        runningAccVali = runningAccVali[:epochs]
                        runningDiceVali = runningDiceVali[:epochs]
                    break

            print("Done")
            if PRINT_TIME:
                endTrain = time.time()
                print("  Training took:", CommonUtil.DecodeSecondToFormatedString(endTrain - startTrain),"in total.")

            # Save model

            dirMdl, filenameMdl = os.path.split(pathModel)
            CommonUtil.Mkdir(dirMdl)
            torch.save(net.state_dict(), pathModel)
            print("Model saved")

            if SAVE_LOSS:
                # Output Running Loss
                X = np.arange(1,epochs+1,1)
                Y = runningLoss
                if VALIDATE:
                    Y1 = runningLossVali
                plt.title("Running Loss")
                plt.plot(X,Y,label="Training Loss")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                dirRL,filenameRL = os.path.split(pathRunningLossPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningLossPlot)
                plt.cla()

                # Output Running Acc
                X = np.arange(1, epochs + 1, 1)
                Y = runningAcc
                if VALIDATE:
                    Y1 = runningAccVali
                plt.title("Running Acc")
                plt.plot(X, Y, label="Training Acc")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Acc")
                plt.legend()
                dirRL, filenameRL = os.path.split(pathRunningAccPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningAccPlot)
                plt.cla()

                # Output Running Acc
                X = np.arange(1, epochs + 1, 1)
                Y = runningAcc
                if VALIDATE:
                    Y1 = runningAccVali
                plt.title("Running Acc")
                plt.plot(X, Y, label="Training Acc")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Acc")
                plt.legend()
                dirRL, filenameRL = os.path.split(pathRunningAccPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningAccPlot)
                plt.cla()

                # Plot Dice
                X = np.arange(1, epochs + 1, 1)
                plt.figure(figsize=(20,10),dpi=72)
                plt.title("Dice")
                plt.subplot(1, 2, 1)
                plt.xlabel("Epoch")
                plt.ylabel("Dice")
                for cls in range(runningDice.shape[1]):
                    plt.plot(X, runningDice[:, cls], label="Class " + str(cls))
                if VALIDATE:
                    for cls in range(runningDiceVali.shape[1]):
                        plt.plot(X, runningDiceVali[:, cls], label="Class Vali " + str(cls))
                plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
                dirRL, filenameRL = os.path.split(pathRunningDicePlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningDicePlot)
                plt.cla()

        if not TRAIN and TEST:
            net.load_state_dict(torch.load(pathModel))

        if TEST:

            # Load training data/Mk training data
            if LOAD_DATA or SAVE_DATA:
                if PRINT_TIME:
                    startLoadTest = time.time()
                print("Loading test set...")
                if not USE_DISK:
                    datasetTest = ImgDataSetMemory()
                    datasetTest.InitFromNpys(dirSaveDataTest, slices=slices, dis=dis, classes=classes)
                else:
                    datasetTest = ImgDataSetDisk()
                    datasetTest.InitFromDir(dirSaveDataTest, slices=slices, dis=dis, classes=classes)
                print("Done")
                if PRINT_TIME:
                    endLoadTest = time.time()
                    print("  Loading test set took:",
                          CommonUtil.DecodeSecondToFormatedString(endLoadTest - startLoadTest))
            else:
                if PRINT_TIME:
                    startMkTest = time.time()
                print("Making test set...")
                if not USE_DISK:
                    datasetTest = ImgDataSetMemory()
                    datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices, dis=dis,
                                             classes=classes,
                                             resize=resize, aug=aug, preproc=preproc)
                else:
                    datasetTest = ImgDataSetDisk()
                    datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], dirSaveDataTest,
                                             slices=slices, dis=dis,
                                             classes=classes, resize=resize, aug=aug, preproc=preproc)
                print("Done")
                if PRINT_TIME:
                    endMkTest = time.time()
                    print("  Making test set took:", CommonUtil.DecodeSecondToFormatedString(endMkTest - startMkTest))

            # Make loader
            print()
            print("Making test loader...")
            loaderTest = data.DataLoader(dataset=datasetTest, batch_size=batchSizeTrain, shuffle=False)
            print("Done")

            if PRINT_TIME:
                startTest = time.time()
            print()
            print("Testing...")

            rateCorrect = 0
            dice = np.zeros(classes)
            sampleCount = 0
            batchCount = 0
            countImg = 0

            if SAVE_OUTPUT and OUTPUT_NII:
                countIW = 0
                countImgInIW = 0
                imgNiiToSave, maskNiiToSave, idxBeg, idxEnd, lenIW = MakeImgsToSave(countIW,datasetTest)

            if SAVE_RA:
                accAll = np.empty(0,dtype=float)
                diceAll = {}
                for c in range(classes):
                    diceAll[c] = np.empty(0,dtype=float)

            # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
            # computations and reduce memory usage.
            with torch.no_grad():
                for i,batch in enumerate(loaderTest):

                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt)) # [IdxInBatch, Channel, H, W]
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt)) # [IdxInBatch, Channel, H, W]
                    # print(masksNP1.shape)
                    masks = masks.to(device)

                    # Get predictions
                    net.eval()
                    outputs = net(inputs)

                    # Acc
                    predicts = torch.zeros_like(outputs)
                    predicts = predicts.scatter(1, torch.max(outputs, 1, keepdim=True).indices, 1)
                    batchAcc = torch.sum(masks == predicts).item() / masks.numel()
                    # print(torch.sum(masks == predicts))
                    # print(torch.sum(masks[:,0,...] == predicts[:,0,...]))
                    # print(np.unique((masks == predicts).cpu().numpy()))
                    # print(masks.numel())
                    # print(masks.shape)

                    # if testcount<32:
                    #     testcount+=1
                    #     testacc +=batchAcc
                    #     testcor += torch.sum(masks == predicts).item()
                    #     testall += masks.numel()
                    # else:
                    #     print(testcor)
                    #     print(testacc/32)
                    #     # print(testcor/testall)
                    #     # print(testall)
                    #     testcount = 1
                    #     testacc = batchAcc
                    #     testcor = torch.sum(masks == predicts).item()
                    #     testall = masks.numel()

                    # print(batchAcc)
                    rateCorrect += batchAcc

                    # Dice
                    for c in range(classes):
                        predictClass = predicts[:, c, ...]
                        maskClass = masks[:, c, ...]
                        dice[c]+=diceCoefAveTorch(predictClass, maskClass)

                    if SAVE_RA:
                        for j in range(inputs.shape[0]):
                            accAll = np.insert(accAll, len(accAll), torch.sum(masks[j] == predicts[j]).item() / masks[j].numel())
                            for c in range(classes):
                                predictClass = predicts[j, c, ...]
                                maskClass = masks[j, c, ...]
                                diceAll[c] = np.insert(diceAll[c], len(diceAll[c]), diceCoefAveTorch(predictClass, maskClass))

                    if SAVE_OUTPUT:
                        CommonUtil.Mkdir(dirTarg)

                        # Output
                        inputsNP = inputs.cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]

                        if not OUTPUT_NII:
                            masksNP = masks.cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                            masksNP1 = np.transpose(masksNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                            masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)

                        predictsNP1 = np.transpose(predicts.cpu().numpy(), (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        predictsNP2 = CommonUtil.UnpackFromOneHot(predictsNP1)

                        if not OUTPUT_NII and OUTPUT_PROB:
                            outputsNP = tnn.Softmax(dim=1)(outputs)[:,1,...].cpu().numpy()
                            outputsNP1 = np.transpose(outputsNP, (1,2,0))  # ndarray [H, W, Channel, IdxInBatch]

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            img = inputsNP1[:,:,int(inputsNP1.shape[2]/2),iImg]
                            img255=ImageProcessor.MapTo255(img)

                            predict = predictsNP2[:,:, iImg]
                            predict255 = ImageProcessor.MapTo255(predict, max=classes-1)

                            if OUTPUT_NII:
                                if countImgInIW<lenIW-1:
                                    # Not enough for one nii
                                    imgNiiToSave[... , countImgInIW+idxBeg] = img255
                                    maskNiiToSave[..., countImgInIW + idxBeg] = predict255
                                    countImgInIW+=1
                                else:
                                    # Enough for one nii
                                    # Save
                                    imgNiiToSave[..., countImgInIW + idxBeg] = img255
                                    maskNiiToSave[..., countImgInIW + idxBeg] = predict255

                                    NiiProcessor.SaveNii(dirTarg,str(countIW)+"_IMG.nii",NiiProcessor.SaveImgsAsNii(imgNiiToSave, niisAll["niisDataTest"][int(countIW/4)]))
                                    NiiProcessor.SaveNii(dirTarg,str(countIW)+"_MASK.nii",NiiProcessor.SaveImgsAsNii(maskNiiToSave, niisAll["niisMaskTest"][int(countIW/4)]))

                                    # New
                                    if countIW<datasetTest.imgDataWrappers.shape[0]-1:
                                        countIW+=1
                                        countImgInIW = 0
                                        imgNiiToSave, maskNiiToSave, idxBeg, idxEnd, lenIW = MakeImgsToSave(countIW,datasetTest)

                            else:
                                ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_ORG.jpg", img255)
                                ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_PRED.jpg", predict255)

                                mask = masksNP2[:,:, iImg]
                                mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                                ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_TARG.jpg", mask255)

                                if OUTPUT_PROB:
                                    output = outputsNP1[:, :, iImg]
                                    output255 = ImageProcessor.MapTo255(output,max=1.0)
                                    ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_OUTPROB.jpg", output255)

                    sampleCount += loaderTest.batch_size
                    batchCount += 1

                    gc.collect()

            print("Done")
            if PRINT_TIME:
                endTest = time.time()
                print("  Testing took:", CommonUtil.DecodeSecondToFormatedString(endTest - startTest),"in total.")

            accuracy = rateCorrect / batchCount
            dice = dice / batchCount

            if SAVE_RA:
                SaveBoxPlot(accAll, os.path.join(dirRAPlot,"Box_Acc.jpg"))
                np.save(os.path.join(dirRAPlot,"testacc.npy"),accAll)
                for c in range(classes):
                    SaveBoxPlot(diceAll[c], os.path.join(dirRAPlot, "Box_Dice"+str(c)+".jpg"))
                    np.save(os.path.join(dirRAPlot, "testdice"+str(c)+".npy"),diceAll[c])

    if PRINT_TIME:
        endProc = time.time()
        print("*"*50)
        print("  Procedure took:", CommonUtil.DecodeSecondToFormatedString(endProc - startProc))

    return accuracy, dice

# Main Framework to run with more than one type of input
# Note: have not yet add output as nii
# Note: basically the same with RunNN but only few differences to adapt to multi-input
# Diff in params:
# types int: types of input to use
# aug func: Input: ndarray[Type,H,W,Slice] fmt=Grey1
#                  ndarray[H,W,Slice] fmt=GreyStep
#           Output: ndarray[Type,CountAug,H,W,Slice] fmt=Grey1
#                   ndarray[CountAug,H,W,Slice] fmt=GreyStep)
# preproc func: Input: ndarray[H,W,Type,Slices] Grey1
#                      ndarray [H,W,Slice] GreyStep
#                      int
#               Output: ndarray[H,W,Type,Slices] Grey1
#                       ndarray [H,W,Slice] GreyStep
# dirsSrcData [type] str: list of directories to read input images from
# dirTarg str: the directory to read input labels from
def RunNNMulti(types, classes, slices, dis, resize,
          aug, preproc,
          trainTestSplit, batchSizeTrain, epochs, learningRate,
          toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput, toSaveResultAnalysis,
          dirSaveData, pathModel, dirsSrcData, dirSrcMask, dirTarg, dirRAPlot, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
          dataFmt="float32", randSeed=0, toPrintTime=True,
          toValidate=True, trainValidationSplit=0.85):
    #
    # Main
    #

    np.random.seed(randSeed)

    SAVE_DATA = toSaveData
    LOAD_DATA = toLoadData
    TRAIN = toTrain
    VALIDATE = toValidate
    TEST = toTest
    SAVE_OUTPUT = toSaveOutput
    SAVE_LOSS = toSaveRunnningLoss
    SAVE_RA = toSaveResultAnalysis
    PRINT_TIME = toPrintTime

    dirTrain = "train"
    dirVali = "vali"
    dirTest = "test"
    dirSaveDataTrain = os.path.join(dirSaveData, dirTrain)
    dirSaveDataVali = os.path.join(dirSaveData, dirVali)
    dirSaveDataTest = os.path.join(dirSaveData, dirTest)

    #   Printing Param
    printLossPerBatch = False

    accuracy = None
    dice = None

    if PRINT_TIME:
        startProc = time.time()

    if ((TRAIN or TEST) and not LOAD_DATA) or SAVE_DATA:
        #
        # Prepare Dataset
        #
        # Load nii

        arrNiisData = []
        for dir in dirsSrcData:
            arrNiisData += [NiiProcessor.ReadAllNiiInDir(dir)]
        niisMask = NiiProcessor.ReadAllNiiInDir(dirSrcMask)

        print()
        print("Niis to read: ")
        print(arrNiisData)
        print(niisMask)
        # Split train set and test set
        if VALIDATE:
            niisAll = ImgDataSetMultiTypesMemory.Split(arrNiisData, niisMask, trainTestSplit, toValidate=True, valiSize=trainValidationSplit)
        else:
            niisAll = ImgDataSetMultiTypesMemory.Split(arrNiisData, niisMask, trainTestSplit)

    if SAVE_DATA:

        # Create and save training data

        if PRINT_TIME:
            startMkTrain = time.time()
        print("Making training set...")

        datasetTrain = ImgDataSetMultiTypesMemory()
        datasetTrain.InitFromNiis(niisAll["arrNiisDataTrain"], niisAll["niisMaskTrain"], slices=slices, dis=dis,
                                  classes=classes, resize=resize, aug=aug, preproc=preproc)

        print("Done")
        if PRINT_TIME:
            endMkTrain = time.time()
            print("  Making training set took:", CommonUtil.DecodeSecondToFormatedString(endMkTrain - startMkTrain))

        if PRINT_TIME:
            startSaveTrainData = time.time()
        print("Saving train data...")
        CommonUtil.Mkdir(dirSaveDataTrain)
        datasetTrain.SaveToNpys(dirSaveDataTrain)
        print("Done")
        if PRINT_TIME:
            endSaveTrainData = time.time()
            print("  Saving train data took:",
                  CommonUtil.DecodeSecondToFormatedString(endSaveTrainData - startSaveTrainData))

        del datasetTrain
        gc.collect()

        if VALIDATE:

            # Create and save validation data

            if PRINT_TIME:
                startMkVali = time.time()
            print("Making validation set...")

            datasetVali = ImgDataSetMultiTypesMemory()
            datasetVali.InitFromNiis(niisAll["arrNiisDataValidate"], niisAll["niisMaskValidate"], slices=slices, dis=dis,
                                      classes=classes, resize=resize, aug=aug, preproc=preproc)
            print("Done")
            if PRINT_TIME:
                endMkVali = time.time()
                print("  Making validation set took:", CommonUtil.DecodeSecondToFormatedString(endMkVali - startMkVali))

            if PRINT_TIME:
                startSaveValiData = time.time()
            print("Saving validation data...")
            CommonUtil.Mkdir(dirSaveDataVali)
            datasetVali.SaveToNpys(dirSaveDataVali)
            print("Done")
            if PRINT_TIME:
                endSaveValiData = time.time()
                print("  Saving validation data took:",
                      CommonUtil.DecodeSecondToFormatedString(endSaveValiData - startSaveValiData))

            del datasetVali
            gc.collect()

        # Create and save test data

        if PRINT_TIME:
            startMkTest = time.time()
        print("Making test set...")

        datasetTest = ImgDataSetMultiTypesMemory()
        datasetTest.InitFromNiis(niisAll["arrNiisDataTest"], niisAll["niisMaskTest"], slices=slices, dis=dis, classes=classes,
                                 resize=resize, aug=aug, preproc=preproc)

        print("Done")
        if PRINT_TIME:
            endMkTest = time.time()
            print("  Making test set took:", CommonUtil.DecodeSecondToFormatedString(endMkTest - startMkTest))

        if PRINT_TIME:
            startSaveTestData = time.time()
        print("Saving test data...")
        CommonUtil.Mkdir(dirSaveDataTest)
        datasetTest.SaveToNpys(dirSaveDataTest)
        print("Done")
        if PRINT_TIME:
            endSaveTestData = time.time()
            print("  Saving test data took:",
                  CommonUtil.DecodeSecondToFormatedString(endSaveTestData - startSaveTestData))

        del datasetTest
        gc.collect()

    if TRAIN or TEST:

        #
        # Choose device
        #
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print()
        print("Device: ", str(device))

        # Create DataLoaders
        # TODO: 使用torchvision?
        # DataLoader return tensor [batch,h,w,c]

        # Prepare Network
        net = UNet_1(16, types*slices, classes, depth=5, inputHW=None, dropoutRate=0.5).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)#UNet_0(slices,classes).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)

        if TRAIN:

            # Load training data / Mk training data
            if LOAD_DATA or SAVE_DATA:
                if PRINT_TIME:
                    startLoadTrain = time.time()
                print()
                print("Loading training set...")

                datasetTrain = ImgDataSetMultiTypesMemory()
                datasetTrain.InitFromNpys(dirSaveDataTrain, slices=slices, dis=dis, classes=classes)

                print("Done")
                if PRINT_TIME:
                    endLoadTrain = time.time()
                    print("  Loading training set took:",
                          CommonUtil.DecodeSecondToFormatedString(endLoadTrain - startLoadTrain))

                if VALIDATE:
                    if PRINT_TIME:
                        startLoadVali = time.time()
                    print()
                    print("Loading validation set...")

                    datasetVali = ImgDataSetMultiTypesMemory()
                    datasetVali.InitFromNpys(dirSaveDataVali, slices=slices, dis=dis, classes=classes)

                    print("Done")
                    if PRINT_TIME:
                        endLoadVali = time.time()
                        print("  Loading validation set took:",
                              CommonUtil.DecodeSecondToFormatedString(endLoadVali - startLoadVali))
            else:
                if PRINT_TIME:
                    startMkTrain = time.time()
                print("Making training set...")

                datasetTrain = ImgDataSetMultiTypesMemory()
                datasetTrain.InitFromNiis(niisAll["arrNiisDataTrain"], niisAll["niisMaskTrain"], slices=slices, dis=dis,
                                          classes=classes, resize=resize, aug=aug, preproc=preproc)

                print("Done")
                if PRINT_TIME:
                    endMkTrain = time.time()
                    print("  Making training set took:",
                          CommonUtil.DecodeSecondToFormatedString(endMkTrain - startMkTrain))

                if VALIDATE:

                    # Create and save validation data

                    if PRINT_TIME:
                        startMkVali = time.time()
                    print("Making validation set...")

                    datasetVali = ImgDataSetMultiTypesMemory()
                    datasetVali.InitFromNiis(niisAll["arrNiisDataValidate"], niisAll["niisMaskValidate"], slices=slices, dis=dis,
                                             classes=classes, resize=resize, aug=aug, preproc=preproc)

                    print("Done")
                    if PRINT_TIME:
                        endMkVali = time.time()
                        print("  Making validation set took:",
                              CommonUtil.DecodeSecondToFormatedString(endMkVali - startMkVali))

            # Make loader
            print()
            print("Making train loader...")
            loaderTrain = data.DataLoader(dataset=datasetTrain, batch_size=batchSizeTrain, shuffle=True)
            print("Done")
            if VALIDATE:
                print()
                print("Making validation loader...")
                loaderVali = data.DataLoader(dataset=datasetVali, batch_size=batchSizeTrain, shuffle=True)
                print("Done")

            # Train

            if PRINT_TIME:
                startTrain = time.time()
            print()
            print("Training...")
            net.apply(weight_init)
            criterion = LossFunc()
            optimiser = topti.Adam(net.parameters(), lr=learningRate)  # Minimise the loss using the Adam algorithm.

            if SAVE_LOSS:
                runningLoss = np.zeros(epochs, dtype=np.float32)
                runningAcc = np.zeros(epochs, dtype=np.float32)
                runningDice = np.zeros((epochs, classes), dtype=np.float32)
                if VALIDATE:
                    runningLossVali = np.zeros(epochs, dtype=np.float32)
                    runningAccVali = np.zeros(epochs, dtype=np.float32)
                    runningDiceVali = np.zeros((epochs, classes), dtype=np.float32)

            global _abort
            _abort = None
            thd = Thread(target=anyKeyToAbort)
            thd.daemon = True
            thd.start()
            print("NOTE: Input anything during training to abort before the next epoch.")

            params = {}

            weightAveLast = {}
            gradAveLast = {}

            # Run epochs
            for epoch in range(epochs):

                # for name, param in net.named_parameters():
                #     if epoch==0:
                #         params[name]=param
                #     else:
                #         print(params[name]-param)
                #         params[name] = param
                #     # print(name, param)

                if PRINT_TIME:
                    startEpoch = time.time()
                print("Running epoch: ",epoch+1)

                # Run validation batch
                if VALIDATE:
                    epoLossVali = 0
                    sampleCountVali = 0
                    batchCountVali = 0
                    accVali = 0
                    diceVali = np.zeros(classes)
                    with torch.no_grad():
                        for i, batch in enumerate(loaderVali):
                            # print("     Validation Batch: ", i+1)
                            inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                            inputs = inputs.to(device)

                            masks = batch[1].type(
                                CommonUtil.PackIntoTorchType(dataFmt))  # Required to be converted from bool to float
                            masks = masks.to(device)

                            net.eval()
                            outputs = net(inputs)

                            # Loss
                            loss = criterion(outputs, masks)
                            ls = loss.item()
                            if printLossPerBatch:
                                print("VALI LOSS: ", ls)
                            epoLossVali += ls

                            # Acc
                            predicts = torch.zeros_like(outputs)
                            predicts = predicts.scatter(1, torch.max(outputs, 1, keepdim=True).indices, 1)
                            accVali += torch.sum(masks == predicts).item() / masks.numel()

                            # Dice
                            for j in range(classes):
                                predictClass = predicts[:, j, ...]
                                maskClass = masks[:, j, ...]
                                diceVali[j] += diceCoefAveTorch(predictClass, maskClass)

                            sampleCountVali += inputs.shape[0]
                            batchCountVali += 1

                            gc.collect()

                epoLoss = 0
                acc = 0
                dice = np.zeros(classes)
                sampleCount = 0
                batchCount = 0

                if DEBUG_SHOW_INPUT:
                    countImg = 0
                    dirINPUTSAVE = "E:/Project/Python/ThesisT1/Sources/DEBUG_INPUT"

                # Train batch
                for i, batch in enumerate(loaderTrain):

                    #print("     Batch: ", i+1)
                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt)) # Required to be converted from bool to float
                    masks = masks.to(device)

                    # masks_ = masks.detach().cpu().numpy()
                    # print(masks_.shape)
                    # print("mask_0", np.unique(masks_[:, 0, ...]))
                    # print("mask_1", np.unique(masks_[:, 1, ...]))

                    if DEBUG_SHOW_INPUT:
                        CommonUtil.Mkdir(dirTarg)

                        # Output
                        inputsNP = inputs.detach().cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]  Channel: [t0s0,t0s1,t0s2,...,t1s0,t1s1,...,...]

                        masksNP = masks.detach().cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        masksNP1 = np.transpose(masksNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            for t in range(types):
                                img = inputsNP1[:,:,t*slices+int(slices/2),iImg]
                                img255=ImageProcessor.MapTo255(img)
                                ImageProcessor.SaveGrayImg(dirINPUTSAVE, str(countImg) + "_ORG_"+str(t)+".jpg", img255)

                            mask = masksNP2[:,:, iImg]
                            mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                            ImageProcessor.SaveGrayImg(dirINPUTSAVE, str(countImg) + "_TARG.jpg", mask255)

                    optimiser.zero_grad()

                    # print("inputs.shape",inputs.shape)
                    net.train()
                    outputs = net(inputs)
                    # print("outputs.shape", outputs.shape)
                    # print("masks.shape", outputs.shape)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimiser.step()

                    # Loss
                    ls = loss.item()
                    if printLossPerBatch:
                        print("LOSS: ", ls)
                    epoLoss += ls

                    # Acc
                    predicts = torch.zeros_like(outputs)
                    predicts = predicts.scatter(1,torch.max(outputs, 1, keepdim=True).indices,1)
                    acc += torch.sum(masks == predicts).item() / masks.numel()

                    # Dice
                    for j in range(classes):
                        predictClass = predicts[:, j, ...]
                        maskClass = masks[:, j, ...]
                        dice[j] += diceCoefAveTorch(predictClass, maskClass)

                    sampleCount+= inputs.shape[0]
                    batchCount += 1

                    # print("    ", "+" * 100)
                    # for name, params in net.named_parameters():
                    #     print(type(params))
                    #     weightAve = torch.mean(params.data)
                    #     gradAve = torch.mean(params.grad)
                    #
                    #     print('     -->name:', name, '-->grad_requirs:', params.requires_grad, '--weight',
                    #           weightAve, ' -->grad_value:', gradAve)
                    #
                    #     if name not in weightAveLast:
                    #         weightAveLast[name] = weightAve
                    #     else:
                    #         print("    -->name:", name, "-->weight diff:",weightAve-weightAveLast[name])
                    #         weightAveLast[name] = weightAve
                    #     gradAveLast[name] = gradAve
                    #     if gradAve==0:
                    #         print("            WARNING: zero_grad!")

                    gc.collect()

                epoLoss = epoLoss / batchCount
                acc = acc / batchCount
                dice = dice / batchCount
                if VALIDATE:
                    epoLossVali = epoLossVali / batchCountVali
                    accVali = accVali / batchCountVali
                    diceVali = diceVali / batchCountVali
                    print("Epoch Summary:")
                    print("  Loss: %f, Loss Vali: %f" % (epoLoss, epoLossVali))
                    print("  Acc: %f, Acc Vali: %f" % (acc, accVali))
                    print("  Dice | Dice Vali:")
                    for cls in range(classes):
                        print("    Class %d: %f | %f" % (cls,dice[cls],diceVali[cls]))
                else:
                    print("Epoch Summary:")
                    print("  Loss: %f" % (epoLoss))
                    print("  Acc: %f" % (acc))
                    print("  Dice | Dice Vali:")
                    for cls in range(classes):
                        print("    Class %d: %f" % (cls, dice[cls]))
                if PRINT_TIME:
                    endEpoch = time.time()
                    print("  Epoch took:", CommonUtil.DecodeSecondToFormatedString(endEpoch - startEpoch))
                print("-" * 30)

                if SAVE_LOSS:
                    runningLoss[epoch] = epoLoss
                    runningAcc[epoch] = acc
                    runningDice[epoch,:]=dice
                    if VALIDATE:
                        runningLossVali[epoch] = epoLossVali
                        runningAccVali[epoch] = accVali
                        runningDiceVali[epoch, :] = diceVali
                if _abort is not None:
                    # Abort
                    epochs = epoch+1
                    runningLoss = runningLoss[:epochs]
                    runningAcc = runningAcc[:epochs]
                    runningDice = runningDice[:epochs]
                    if VALIDATE:
                        runningLossVali = runningLossVali[:epochs]
                        runningAccVali = runningAccVali[:epochs]
                        runningDiceVali = runningDiceVali[:epochs]
                    break

            print("Done")
            if PRINT_TIME:
                endTrain = time.time()
                print("  Training took:", CommonUtil.DecodeSecondToFormatedString(endTrain - startTrain),"in total.")

            # Save model

            dirMdl, filenameMdl = os.path.split(pathModel)
            CommonUtil.Mkdir(dirMdl)
            torch.save(net.state_dict(), pathModel)
            print("Model saved")

            if SAVE_LOSS:
                # Output Running Loss
                X = np.arange(1,epochs+1,1)
                Y = runningLoss
                if VALIDATE:
                    Y1 = runningLossVali
                plt.title("Running Loss")
                plt.plot(X,Y,label="Training Loss")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                dirRL,filenameRL = os.path.split(pathRunningLossPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningLossPlot)
                plt.cla()

                # Output Running Acc
                X = np.arange(1, epochs + 1, 1)
                Y = runningAcc
                if VALIDATE:
                    Y1 = runningAccVali
                plt.title("Running Acc")
                plt.plot(X, Y, label="Training Acc")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Acc")
                plt.legend()
                dirRL, filenameRL = os.path.split(pathRunningAccPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningAccPlot)
                plt.cla()

                # Output Running Acc
                X = np.arange(1, epochs + 1, 1)
                Y = runningAcc
                if VALIDATE:
                    Y1 = runningAccVali
                plt.title("Running Acc")
                plt.plot(X, Y, label="Training Acc")
                if VALIDATE:
                    plt.plot(X, Y1, label="Validation Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Acc")
                plt.legend()
                dirRL, filenameRL = os.path.split(pathRunningAccPlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningAccPlot)
                plt.cla()

                # Plot Dice
                X = np.arange(1, epochs + 1, 1)
                plt.figure(figsize=(20,10),dpi=72)
                plt.title("Dice")
                plt.subplot(1, 2, 1)
                plt.xlabel("Epoch")
                plt.ylabel("Dice")
                for cls in range(runningDice.shape[1]):
                    plt.plot(X, runningDice[:, cls], label="Class " + str(cls))
                if VALIDATE:
                    for cls in range(runningDiceVali.shape[1]):
                        plt.plot(X, runningDiceVali[:, cls], label="Class Vali " + str(cls))
                plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
                dirRL, filenameRL = os.path.split(pathRunningDicePlot)
                CommonUtil.Mkdir(dirRL)
                plt.savefig(pathRunningDicePlot)
                plt.cla()

        if not TRAIN and TEST:
            net.load_state_dict(torch.load(pathModel))

        if TEST:

            # Load training data/Mk training data

            if LOAD_DATA or SAVE_DATA:
                if PRINT_TIME:
                    startLoadTest = time.time()
                print("Loading test set...")

                datasetTest = ImgDataSetMultiTypesMemory()
                datasetTest.InitFromNpys(dirSaveDataTest, slices=slices, dis=dis, classes=classes)

                print("Done")
                if PRINT_TIME:
                    endLoadTest = time.time()
                    print("  Loading test set took:",
                          CommonUtil.DecodeSecondToFormatedString(endLoadTest - startLoadTest))
            else:
                if PRINT_TIME:
                    startMkTest = time.time()
                print("Making test set...")

                datasetTest = ImgDataSetMultiTypesMemory()
                datasetTest.InitFromNiis(niisAll["arrNiisDataTest"], niisAll["niisMaskTest"], slices=slices, dis=dis, classes=classes,
                                         resize=resize, aug=aug, preproc=preproc)

                print("Done")
                if PRINT_TIME:
                    endMkTest = time.time()
                    print("  Making test set took:", CommonUtil.DecodeSecondToFormatedString(endMkTest - startMkTest))

            # Make loader
            print()
            print("Making test loader...")
            loaderTest = data.DataLoader(dataset=datasetTest, batch_size=batchSizeTrain, shuffle=False)
            print("Done")

            if PRINT_TIME:
                startTest = time.time()
            print()
            print("Testing...")

            rateCorrect = 0
            dice = np.zeros(classes)
            sampleCount = 0
            batchCount = 0
            countImg = 0

            if SAVE_RA:
                accAll = np.empty(0,dtype=float)
                diceAll = {}
                for c in range(classes):
                    diceAll[c] = np.empty(0,dtype=float)

            # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
            # computations and reduce memory usage.
            with torch.no_grad():
                for i,batch in enumerate(loaderTest):

                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt))
                    # print(masksNP1.shape)
                    masks = masks.to(device)

                    # Get predictions
                    net.eval()
                    outputs = net(inputs)

                    # Acc
                    predicts = torch.zeros_like(outputs)
                    predicts = predicts.scatter(1, torch.max(outputs, 1, keepdim=True).indices, 1)
                    batchAcc = torch.sum(masks == predicts).item() / masks.numel()
                    # print(torch.sum(masks == predicts))
                    # print(torch.sum(masks[:,0,...] == predicts[:,0,...]))
                    # print(np.unique((masks == predicts).cpu().numpy()))
                    # print(masks.numel())
                    # print(masks.shape)

                    # if testcount<32:
                    #     testcount+=1
                    #     testacc +=batchAcc
                    #     testcor += torch.sum(masks == predicts).item()
                    #     testall += masks.numel()
                    # else:
                    #     print(testcor)
                    #     print(testacc/32)
                    #     # print(testcor/testall)
                    #     # print(testall)
                    #     testcount = 1
                    #     testacc = batchAcc
                    #     testcor = torch.sum(masks == predicts).item()
                    #     testall = masks.numel()

                    # print(batchAcc)
                    rateCorrect += batchAcc


                    # Dice
                    for j in range(classes):
                        predictClass = predicts[:, j, ...]
                        maskClass = masks[:, j, ...]
                        dice[j]+=diceCoefAveTorch(predictClass, maskClass)

                    if SAVE_RA:
                        for j in range(inputs.shape[0]):
                            accAll = np.insert(accAll, len(accAll), torch.sum(masks[j] == predicts[j]).item() / masks[j].numel())
                            for c in range(classes):
                                predictClass = predicts[j, c, ...]
                                maskClass = masks[j, c, ...]
                                diceAll[c] = np.insert(diceAll[c], len(diceAll[c]), diceCoefAveTorch(predictClass, maskClass))

                    if SAVE_OUTPUT:
                        CommonUtil.Mkdir(dirTarg)

                        # Output
                        inputsNP = inputs.cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]

                        masksNP = masks.cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        masksNP1 = np.transpose(masksNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)

                        predictsNP1 = np.transpose(predicts.cpu().numpy(), (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        predictsNP2 = CommonUtil.UnpackFromOneHot(predictsNP1)

                        if OUTPUT_PROB:
                            outputsNP = tnn.Softmax(dim=1)(outputs)[:,1,...].cpu().numpy()
                            outputsNP1 = np.transpose(outputsNP, (1,2,0))  # ndarray [H, W, Channel, IdxInBatch]

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            for t in range(types):
                                img = inputsNP1[:,:,t*slices+int(slices/2),iImg]
                                img255=ImageProcessor.MapTo255(img)
                                ImageProcessor.SaveGrayImg(dirINPUTSAVE, str(countImg) + "_ORG_"+str(t)+".jpg", img255)

                            mask = masksNP2[:,:, iImg]
                            mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(mask255, "P" + str(iImg)+"_TARG", (10, 10))
                            ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_TARG.jpg", mask255)

                            predict = predictsNP2[:,:, iImg]
                            predict255 = ImageProcessor.MapTo255(predict, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(predict255, "P" + str(iImg)+"_PRED", (10, 10))
                            ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_PRED.jpg", predict255)

                            if OUTPUT_PROB:
                                output = outputsNP1[:, :, iImg]
                                output255 = ImageProcessor.MapTo255(output,max=1.0)
                                ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_OUTPROB.jpg", output255)

                    sampleCount += loaderTest.batch_size
                    batchCount += 1

                    gc.collect()

            print("Done")
            if PRINT_TIME:
                endTest = time.time()
                print("  Testing took:", CommonUtil.DecodeSecondToFormatedString(endTest - startTest),"in total.")

            accuracy = rateCorrect / batchCount
            dice = dice / batchCount

            if SAVE_RA:
                SaveBoxPlot(accAll, os.path.join(dirRAPlot,"Box_Acc.jpg"))
                np.save(os.path.join(dirRAPlot,"testacc.npy"),accAll)
                for c in range(classes):
                    SaveBoxPlot(diceAll[c], os.path.join(dirRAPlot, "Box_Dice"+str(c)+".jpg"))
                    np.save(os.path.join(dirRAPlot, "testdice"+str(c)+".npy"),diceAll[c])

    if PRINT_TIME:
        endProc = time.time()
        print("*"*50)
        print("  Procedure took:", CommonUtil.DecodeSecondToFormatedString(endProc - startProc))

    return accuracy, dice

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

# Run net for real data
def Main():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toSaveData = False
    toLoadData = True
    toTrain = False
    toTest = True
    toSaveRunnningLoss = True
    toSaveOutput = True
    toSaveResultAnalysis = True

    #
    # Param Setting
    #
    # Running Params
    classes = 2
    trainTestSplit = 0.8 #0.9
    batchSizeTrain = 32 #24
    slices = 1
    dis = 1
    resize = None #(256, 256)
    epochs = 50 #30 #250
    learningRate = 0.0005 #0.0001 #0.004
    dataFmt = "float32"
    toUseDisk = False

    #
    # Set all kinds of paths
    #
    dirSrc = "../../../Sources/Data/data_nii"

    dirRoot = "../../../Sources/T4C2_HalfData"
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    dirRAPlot = dirRoot
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")
    pathRunningAccPlot = os.path.join(dirRoot,"acc.jpg")
    pathRunningDicePlot = os.path.join(dirRoot,"dice.jpg")

    accuracy, dice = RunNN(classes, slices, dis, resize,
                           DataAug, PreprocT4,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput, toSaveResultAnalysis,
                           dirSaveData, pathModel, dirSrc, dirTarg, dirRAPlot, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
                           toUseDisk, dataFmt, randSeed,
                           toValidate=True, trainValidationSplit=0.85) #trainValidationSplit=0.95

    #
    # Print result
    #
    print("Classification accuracy: ", accuracy)

    print("Dice Coef:")

    for j in range(classes):
        print("Class ", j, ": ", dice[j])

# Run net for real data with multi-type input
def MainMT():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toSaveData = False
    toLoadData = True
    toTrain = False
    toTest = True
    toSaveRunnningLoss = True
    toSaveOutput = False
    toSaveResultAnalysis = True

    #
    # Param Setting
    #
    #   Running Params
    types = 2
    classes = 2
    trainTestSplit = 0.8 #0.9
    batchSizeTrain = 32 #24
    slices = 1
    dis = 1
    resize = None #(256, 256)
    epochs = 50 #250
    learningRate = 0.0005 #0.0001 #0.004
    dataFmt = "float32"

    dirsSrcData = ["../../../Sources/Data/data_nii/dataBkup/T1","../../../Sources/Data/data_nii/dataBkup/T4"] #["../../../Sources/Data/data_nii/dataBkup/ljpt","../../../Sources/Data/data_nii/dataBkup/ljpt1"]
    dirSrcMask = "../../../Sources/Data/data_nii/masks" #"../../../Sources/Data/data_nii/maskLJPT"

    dirRoot = "../../../Sources/T2T4C2_HalfData"
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    dirRAPlot = dirRoot
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")
    pathRunningAccPlot = os.path.join(dirRoot,"acc.jpg")
    pathRunningDicePlot = os.path.join(dirRoot,"dice.jpg")

    accuracy, dice = RunNNMulti(types, classes, dis, slices, resize,
                           DataAugMultiNiis, PreprocMultiNiis,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput, toSaveResultAnalysis,
                           dirSaveData, pathModel, dirsSrcData, dirSrcMask, dirTarg, dirRAPlot, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
                           dataFmt=dataFmt, randSeed=randSeed,
                           toValidate=True, trainValidationSplit=0.85) #trainValidationSplit=0.95

    print("Classification accuracy: ", accuracy)

    print("Dice Coef:")

    for j in range(classes):
        print("Class ", j, ": ", dice[j])

# Run net for real data
# Using disk to store data instead of memory
def Main_MEM_SAVE():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toSaveData = False
    toLoadData = True
    toTrain = True
    toTest = True
    toSaveRunnningLoss = True
    toSaveOutput = True

    #
    # Param Setting
    #
    #   Running Params
    classes = 2
    trainTestSplit = 0.8
    batchSizeTrain = 32
    slices = 3
    resize = None #(256, 256)
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"
    toUseDisk = True

    dirSrc = "../../../Sources/Data/data_nii"

    dirRoot = "../../../Sources/T4C2"
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")
    pathRunningAccPlot = os.path.join(dirRoot,"acc.jpg")
    pathRunningDicePlot = os.path.join(dirRoot,"dice.jpg")

    accuracy, dice = RunNN(classes, slices, resize,
                           DataAug, PreprocT4,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                           dirSaveData, pathModel, dirSrc, dirTarg, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
                           toUseDisk, dataFmt, randSeed,
                           toValidate=True, trainValidationSplit=0.85)

    print("Classification accuracy: ", accuracy)

    print("Dice Coef:")

    for j in range(classes):
        print("Class ", j, ": ", dice[j])

# Only used to restore some data
def RESTORE_MISSING_DATA():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toSaveData = True
    toLoadData = False
    toTrain = False
    toTest = False
    toSaveRunnningLoss = False
    toSaveOutput = False
    toSaveResultAnalysis = False


    #
    # Param Setting
    #
    #   Running Params
    classes = 2
    trainTestSplit = 0
    batchSizeTrain = 8
    slices = 1
    dis = 1
    resize = None #(256, 256)
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"
    toUseDisk = False

    dirSrc = "../../../Sources/Data/data_nii"

    dirRoot = "../../../Sources/Restore"
    dirRAPlot = dirRoot
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")
    pathRunningAccPlot = os.path.join(dirRoot, "acc.jpg")
    pathRunningDicePlot = os.path.join(dirRoot, "dice.jpg")

    accuracy, dice = RunNN(classes, slices, dis, resize,
                           DataAug, PreprocT4,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput, toSaveResultAnalysis,
                           dirSaveData, pathModel, dirSrc, dirTarg, dirRAPlot, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
                           toUseDisk, dataFmt, randSeed,
                           toValidate=False, trainValidationSplit=0.85)

if __name__ == '__main__':
    Main()
    #MainMT()
    #Main_MEM_SAVE()

    #RESTORE_MISSING_DATA()

    #TestNetwork()
