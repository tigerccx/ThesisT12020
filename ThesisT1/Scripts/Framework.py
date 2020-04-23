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
from DataStructures import ImgDataWrapperDisk
from DataStructures import ImgDataWrapperMemory

from DataAugAndPreProc import PreprocDistBG_TRIAL
from DataAugAndPreProc import Preproc0_TRIAL
from DataAugAndPreProc import PreprocT4
from DataAugAndPreProc import DataAug

from Network import UNet_0
from Network import UNet_1

from LossFunc import MulticlassDiceLoss

#MACRO
DEBUG = False
DEBUG_TEST = False

# Abort
_abort = None
def anyKeyToAbort():
    global _abort
    _abort = input()

def weight_init(m):
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, tnn.Conv2d) or isinstance(m, tnn.ConvTranspose2d):
        tnn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
     # 是否为批归一化层
    elif isinstance(m, tnn.BatchNorm2d):
        tnn.init.constant_(m.weight, 1)
        tnn.init.constant_(m.bias, 0)

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

    return loss.item()

# Input: Tensor [IdxInBatch, H, W]
#        Tensor [IdxInBatch, H, W]
# Output: float diceCoef
def diceCoefTorch(input, target):
    N = target.shape[0]
    smooth = 1

    input_flat = input.view((N, -1))
    target_flat = target.view((N, -1))

    intersection = input_flat * target_flat

    loss = torch.sum((2 * (torch.sum(intersection,1)) + smooth) / (torch.sum(input_flat,1) + torch.sum(target_flat,1) + smooth)) / N

    return loss.item()


#TODO: Add Temp for processed files
#TODO: Add augmentationn
#TODO: Try transfer learning
#TODO: Use multi-thread for reading annd saving

def RunNN(classes, slices, resize,
          aug, preproc,
          trainTestSplit, batchSizeTrain, epochs, learningRate,
          toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
          dirSaveData, pathModel, dirSrc, dirTarg, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
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

    if ((TRAIN or TEST) and not LOAD_DATA) or SAVE_DATA:
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
            datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices,
                                      classes=classes, resize=resize, aug=aug, preproc=preproc)
        else:
            datasetTrain = ImgDataSetDisk()
            datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], dirSaveDataTrain,
                                      slices=slices,
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
                datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], slices=slices,
                                          classes=classes, resize=resize, aug=aug, preproc=preproc)
            else:
                datasetVali = ImgDataSetDisk()
                datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], dirSaveDataVali,
                                          slices=slices, classes=classes, resize=resize, aug=aug, preproc=preproc)

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
            datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices,
                                     classes=classes,
                                     resize=resize, aug=aug, preproc=preproc)
        else:
            datasetTest = ImgDataSetDisk()
            datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], dirSaveDataTest,
                                     slices=slices,
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
        net = UNet_1(8, slices, classes, depth=5, inputHW=None, dropoutRate=0.5).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)#UNet_0(slices,classes).type(CommonUtil.PackIntoTorchType(dataFmt)).to(device)

        if TRAIN:

            # Load training data / Mk training data
            if LOAD_DATA or SAVE_DATA:
                if PRINT_TIME:
                    startLoadTrain = time.time()
                print()
                print("Loading training set...")
                if not USE_DISK:
                    datasetTrain = ImgDataSetMemory()
                    datasetTrain.InitFromNpys(dirSaveDataTrain, slices=slices, classes=classes)
                else:
                    datasetTrain = ImgDataSetDisk()
                    datasetTrain.InitFromDir(dirSaveDataTrain, slices=slices, classes=classes)
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
                        datasetVali.InitFromNpys(dirSaveDataVali, slices=slices, classes=classes)
                    else:
                        datasetVali = ImgDataSetDisk()
                        datasetVali.InitFromDir(dirSaveDataVali, slices=slices, classes=classes)
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
                    datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices,
                                              classes=classes, resize=resize, aug=aug, preproc=preproc)
                else:
                    datasetTrain = ImgDataSetDisk()
                    datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], dirSaveDataTrain,
                                              slices=slices, classes=classes, resize=resize, aug=aug, preproc=preproc)

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
                                                 slices=slices, classes=classes, resize=resize, aug=aug, preproc=preproc)
                    else:
                        datasetVali = ImgDataSetDisk()
                        datasetVali.InitFromNiis(niisAll["niisDataValidate"], niisAll["niisMaskValidate"], dirSaveDataVali,
                                                 slices=slices, classes=classes, resize=resize, aug=aug, preproc=preproc)

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
                epoLoss = 0
                acc = 0
                dice = np.zeros(classes)

                batchCount = 0

                # Run batch
                for i, batch in enumerate(loaderTrain):
                    #print("     Batch: ", i+1)
                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt)) # Required to be converted from bool to float
                    masks = masks.to(device)

                    optimiser.zero_grad()

                    outputs = net(inputs)
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
                        dice[j] += diceCoefTorch(predictClass, maskClass)

                    batchCount+=1

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

                # Run validation batch
                if VALIDATE:
                    epoLossVali = 0
                    batchCountVali = 0
                    accVali = 0
                    diceVali = np.zeros(classes)
                    with torch.no_grad():
                        for i, batch in enumerate(loaderVali):
                            # print("     Validation Batch: ", i+1)
                            inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))
                            inputs = inputs.to(device)

                            masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt))  # Required to be converted from bool to float
                            masks = masks.to(device)

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
                                diceVali[j] += diceCoefTorch(predictClass, maskClass)

                            batchCountVali += 1

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
                    datasetTest.InitFromNpys(dirSaveDataTest, slices=slices, classes=classes)
                else:
                    datasetTest = ImgDataSetDisk()
                    datasetTest.InitFromDir(dirSaveDataTest, slices=slices, classes=classes)
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
                    datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices,
                                             classes=classes,
                                             resize=resize, aug=aug, preproc=preproc)
                else:
                    datasetTest = ImgDataSetDisk()
                    datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], dirSaveDataTest,
                                             slices=slices,
                                             classes=classes, resize=resize, aug=aug, preproc=preproc)
                print("Done")
                if PRINT_TIME:
                    endMkTest = time.time()
                    print("  Making test set took:", CommonUtil.DecodeSecondToFormatedString(endMkTest - startMkTest))

            # Make loader
            print()
            print("Making test loader...")
            loaderTest = data.DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)
            print("Done")

            if PRINT_TIME:
                startTest = time.time()
            print()
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

                    inputs = batch[0].type(CommonUtil.PackIntoTorchType(dataFmt))

                    inputsNP = inputs.numpy() #ndarray [IdxInBatch, Channel, H, W]
                    inputsNP1 = np.transpose(inputsNP, (2, 3, 1, 0)) #ndarray [H, W, Channel, IdxInBatch]

                    inputs = inputs.to(device)

                    masks = batch[1].type(CommonUtil.PackIntoTorchType(dataFmt))

                    # print(masksNP1.shape)
                    masks = masks.to(device)

                    # Get predictions
                    outputs = net(inputs)

                    # Acc
                    predicts = torch.zeros_like(outputs)
                    predicts = predicts.scatter(1, torch.max(outputs, 1, keepdim=True).indices, 1)
                    rateCorrect += torch.sum(masks == predicts).item() / masks.numel()

                    # Dice
                    for j in range(classes):
                        predictClass = predicts[:, j, ...]
                        maskClass = masks[:, j, ...]
                        dice[j]+=diceCoefTorch(predictClass, maskClass)

                    if SAVE_OUTPUT:
                        CommonUtil.Mkdir(dirTarg)

                        # Output
                        masksNP = masks.cpu().numpy()  # ndarray [IdxInBatch, Channel, H, W]
                        masksNP1 = np.transpose(masksNP, (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        masksNP2 = CommonUtil.UnpackFromOneHot(masksNP1)

                        predictsNP1 = np.transpose(predicts.cpu().numpy(), (2, 3, 1, 0))  # ndarray [H, W, Channel, IdxInBatch]
                        predictsNP2 = CommonUtil.UnpackFromOneHot(predictsNP1)

                        for iImg in range(inputsNP1.shape[3]):
                            countImg+=1

                            img = inputsNP1[:,:,int(inputsNP1.shape[2]/2),iImg]
                            img255=ImageProcessor.MapTo255(img)
                            # ImageProcessor.ShowGrayImgHere(img255, "P"+str(iImg),(10,10))
                            ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_ORG.jpg", img255)

                            mask = masksNP2[:,:, iImg]
                            mask255 = ImageProcessor.MapTo255(mask, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(mask255, "P" + str(iImg)+"_TARG", (10, 10))
                            ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_TARG.jpg", mask255)

                            predicts = predictsNP2[:,:, iImg]
                            predict255 = ImageProcessor.MapTo255(predicts, max=classes-1)
                            # ImageProcessor.ShowGrayImgHere(predict255, "P" + str(iImg)+"_PRED", (10, 10))
                            ImageProcessor.SaveGrayImg(dirTarg, str(countImg) + "_PRED.jpg", predict255)

                    gc.collect()

            print("Done")
            if PRINT_TIME:
                endTest = time.time()
                print("  Testing took:", CommonUtil.DecodeSecondToFormatedString(endTest - startTest),"in total.")

            accuracy = rateCorrect / countBatch
            dice = dice / countBatch

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

# Run net for trial data
# NOTE: Need to change RunNN
def Main_TRIAL():
    # Rand Seed
    randSeed = 2

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
    classes = 7
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"
    toUseDisk = False

    dirSrc = "../../../Sources/Data/data_nii_TRIAL"

    dirRoot = "../../../Sources/C7"
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")
    pathRunningAccPlot = os.path.join(dirRoot, "acc.jpg")
    pathRunningDicePlot = os.path.join(dirRoot, "dice.jpg")

    accuracy, dice = RunNN(classes, slices, resize,
                           DataAug, PreprocDistBG_TRIAL,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                           dirSaveData, pathModel, dirSrc, dirTarg, pathRunningLossPlot, pathRunningAccPlot, pathRunningDicePlot,
                           toUseDisk, dataFmt, randSeed,
                           toValidate=True, trainValidationSplit=0.85)

    print("Classification accuracy: ", accuracy)

    print("Dice Coef:")

    for j in range(classes):
        print("Class ", j, ": ", dice[j])

# Run net for real data
def Main():
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
    epochs = 100
    learningRate = 0.0005 #0.004
    dataFmt = "float32"
    toUseDisk = False

    dirSrc = "../../../Sources/Data/data_nii"

    dirRoot = "../../../Sources/T4C2_HalfData"
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

# NOTE: Need to change RunNN
def RESTORE_MISSING_DATA():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toSaveData = False
    toLoadData = False
    toTrain = False
    toTest = True
    toSaveRunnningLoss = False
    toSaveOutput = False

    #
    # Param Setting
    #
    #   Running Params
    classes = 2
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = None #(256, 256)
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"
    toUseDisk = True

    dirSrc = "../../../Sources/Data/data_nii"

    dirRoot = "../../../Sources/Restore"
    dirSaveData = os.path.join(dirRoot,"SavedData")
    pathModel = os.path.join(dirRoot,"model.pth")
    dirTarg = os.path.join(dirRoot,"Output")
    pathRunningLossPlot = os.path.join(dirRoot,"loss.jpg")

    accuracy, dice = RunNN(classes, slices, resize,
                           DataAug, PreprocT4,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toSaveData, toLoadData, toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                           dirSaveData, pathModel, dirSrc, dirTarg, pathRunningLossPlot,
                           toUseDisk, dataFmt, randSeed)

# Test train split compare
# NOTE: NOT YET RUNNABLE!!!
def Main0():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toTrain = True
    toTest = True
    toSaveRunnningLoss = True
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

    dirModel = "../../../Sources/Net/CompareTrainSplit"
    dirLoss = "../../../Sources/Data/loss/CompareTrainSplit"

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
            fnModel = "model_Split"+str(i)+"_Run"+str(count) + ".pth"
            fnLoss = "loss_Split"+str(i)+"_Run"+str(count) + ".jpg"
            pathModel = os.path.join(dirModel, fnModel)
            pathRunningLossPlot = os.path.join(dirLoss, fnLoss)

            print("\n")
            print("*" * 44)
            print("Run: ", count)
            trainTestSplit = XRate[i]
            accuracy, dice = RunNN(classes, slices, resize,
                                   DataAug, PreprocDistBG,
                                   trainTestSplit, batchSizeTrain, epochs, learningRate,
                                   toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                                   pathModel, pathSrc, pathTarg, pathRunningLossPlot,
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

# Test dist or not dist BG
# NOTE: NOT YET RUNNABLE!!!
def Main1():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toTrain = True
    toTest = True
    toSaveRunnningLoss = True
    toSaveOutput = False

    #
    # Param Setting
    #
    #   Running Params
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"

    pathSrc = "../../../Sources/Data/data_nii"
    pathTarg = "../../../Sources/Data/output_test"  # "../../../Sources/Data/output_D2"#"../../../Sources/Data/output"#"../../../Sources/Data/output_6"


    countRun = 3

    classesNDBG = 6
    accNDBG = np.zeros(countRun)
    dicesNDBG = np.zeros((countRun, classesNDBG))
    classes = 7
    acc = np.zeros(countRun)
    dices = np.zeros((countRun, classes))

    # NDBG
    for i in range(countRun):
        pathModel = "../../../Sources/Net/PredictingAllLabels/model_C6_" + str(i) + ".pth"
        pathRunningLossPlot = "../../../Sources/Data/loss/PredictingAllLabels/loss_C6_" + str(i) + ".jpg"
        accuracy, dice = RunNN(classesNDBG, slices, resize,
                               None, Preproc0,
                               trainTestSplit, batchSizeTrain, epochs, learningRate,
                               toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                               pathModel, pathSrc, pathTarg, pathRunningLossPlot,
                               dataFmt, randSeed)

        print("Classification accuracy: ", accuracy)

        print("Dice Coef:")

        for j in range(classesNDBG):
            print("Class ", j, ": ", dice[j])

        accNDBG[i] = accuracy
        for j in range(classesNDBG):
            dicesNDBG[i, j] = dice[j]

    # Usual
    for i in range(countRun):
        pathModel = "../../../Sources/Net/PredictingAllLabels/model_C7_" + str(i) + ".pth"
        pathRunningLossPlot = "../../../Sources/Data/loss/PredictingAllLabels/loss_C7_" + str(i) + ".jpg"
        accuracy, dice = RunNN(classes, slices, resize,
                               None, PreprocDistBG,
                               trainTestSplit, batchSizeTrain, epochs, learningRate,
                               toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                               pathModel, pathSrc, pathTarg, pathRunningLossPlot,
                               dataFmt, randSeed)

        print("Classification accuracy: ", accuracy)

        print("Dice Coef:")

        for j in range(classes):
            print("Class ", j, ": ", dice[j])

        acc[i] = accuracy
        for j in range(classes):
            dices[i, j] = dice[j]

    print("*"*66)
    print("NDBG:")
    print("Classification accuracy: ", accNDBG)

    print("Dice Coef:")
    for j in range(classesNDBG):
        print("Class ", j, ": ", dicesNDBG[:, j])

    print("Acc Ave: ", np.sum(accNDBG)/countRun)
    print("Dice Ave:")
    for j in range(classesNDBG):
        print("Class ", j, ": ", np.sum(dicesNDBG[:, j])/countRun)


    print("*" * 66)
    print("Usual:")
    print("Classification accuracy: ", acc)

    print("Dice Coef:")
    for j in range(classes):
        print("Class ", j, ": ", dices[:, j])

    print("Acc Ave: ", np.sum(acc) / countRun)
    print("Dice Ave:")
    for j in range(classes):
        print("Class ", j, ": ", np.sum(dices[:, j]) / countRun)

# Test loss print
# NOTE: NOT YET RUNNABLE!!!
def Main2():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toTrain = True
    toSaveRunnningLoss = True
    toTest = True
    toSaveOutput = True  # True

    #
    # Param Setting
    #
    #   Running Params
    classes = 6
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"

    pathModel = "./model_C6.pth"  # "./model_D2.pth"#"./model.pth"#"./model_6.pth"
    pathSrc = "../../../Sources/Data/data_nii"
    pathTarg = "../../../Sources/Data/output_C6"  # "../../../Sources/Data/output_D2"#"../../../Sources/Data/output"#"../../../Sources/Data/output_6"
    pathRunningLossPlot = "../../../Sources/Data/loss/loss_C6.jpg"

    accuracy, dice = RunNN(classes, slices, resize,
                           None, Preproc0,
                           trainTestSplit, batchSizeTrain, epochs, learningRate,
                           toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                           pathModel, pathSrc, pathTarg, pathRunningLossPlot,
                           dataFmt, randSeed)

    print("Classification accuracy: ", accuracy)

    print("Dice Coef:")

    for j in range(classes):
        print("Class ", j, ": ", dice[j])

# Test marking only target class (Distinguish BG and not)
# NOTE: NOT YET RUNNABLE!!!
def Main3():
    # Rand Seed
    randSeed = 0

    # Train or Test
    toTrain = True
    toTest = True
    toSaveRunnningLoss = True
    toSaveOutput = True

    #
    # Param Setting
    #
    #   Running Params
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None
    epochs = 50
    learningRate = 0.001
    dataFmt = "float32"

    pathSrc = "../../../Sources/Data/data_nii"
    pathTarg = "../../../Sources/Data/output_test_0"

    countRun = 3

    classesNDBG = 2
    accNDBG = np.zeros(countRun)
    dicesNDBG = np.zeros((countRun, classesNDBG))
    classes = 3
    acc = np.zeros(countRun)
    dices = np.zeros((countRun, classes))

    # NDBG
    for i in range(countRun):
        pathModel = "../../../Sources/Net/PredictingOnlyTarget/model_C2_"+str(i)+".pth"
        pathRunningLossPlot = "../../../Sources/Data/loss/PredictingOnlyTarget/loss_C2_"+str(i)+".jpg"
        accuracy, dice = RunNN(classesNDBG, slices, resize,
                               None, Preproc0,
                               trainTestSplit, batchSizeTrain, epochs, learningRate,
                               toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                               pathModel, pathSrc, pathTarg, pathRunningLossPlot,
                               dataFmt, randSeed)

        print("Classification accuracy: ", accuracy)

        print("Dice Coef:")

        for j in range(classesNDBG):
            print("Class ", j, ": ", dice[j])

        accNDBG[i] = accuracy
        for j in range(classesNDBG):
            dicesNDBG[i, j] = dice[j]

    # Usual
    for i in range(countRun):
        pathModel = "../../../Sources/Net/PredictingOnlyTarget/model_C3_" + str(i) + ".pth"
        pathRunningLossPlot = "../../../Sources/Data/loss/PredictingOnlyTarget/loss_C3_" + str(i) + ".jpg"
        accuracy, dice = RunNN(classes, slices, resize,
                               None, PreprocDistBG,
                               trainTestSplit, batchSizeTrain, epochs, learningRate,
                               toTrain, toSaveRunnningLoss, toTest, toSaveOutput,
                               pathModel, pathSrc, pathTarg, pathRunningLossPlot,
                               dataFmt, randSeed)

        print("Classification accuracy: ", accuracy)

        print("Dice Coef:")

        for j in range(classes):
            print("Class ", j, ": ", dice[j])

        acc[i] = accuracy
        for j in range(classes):
            dices[i, j] = dice[j]

    print("*" * 66)
    print("NDBG:")
    print("Classification accuracy: ", accNDBG)

    print("Dice Coef:")
    for j in range(classesNDBG):
        print("Class ", j, ": ", dicesNDBG[:, j])

    print("Acc Ave: ", np.sum(accNDBG) / countRun)
    print("Dice Ave:")
    for j in range(classesNDBG):
        print("Class ", j, ": ", np.sum(dicesNDBG[:, j]) / countRun)

    print("*" * 66)
    print("Usual:")
    print("Classification accuracy: ", acc)

    print("Dice Coef:")
    for j in range(classes):
        print("Class ", j, ": ", dices[:, j])

    print("Acc Ave: ", np.sum(acc) / countRun)
    print("Dice Ave:")
    for j in range(classes):
        print("Class ", j, ": ", np.sum(dices[:, j]) / countRun)

if __name__ == '__main__':
    #Main_MEM_SAVE()
    #Main_TRIAL()
    # Main()
    #RESTORE_MISSING_DATA()
    #Main0()
    #Main1()
    #Main2()
    #Main3()
    # TestNiiWrapper()
    #TestImgDataSet()
    #TestNetwork()
