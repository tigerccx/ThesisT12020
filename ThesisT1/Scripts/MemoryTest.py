import os
import numpy as np

import torch.utils.data as data

from memory_profiler import profile

from Utils import NiiProcessor
from Utils import ImageProcessor
from Utils import CommonUtil

from DataStructures import ImgDataSet
from DataStructures import ImgDataWrapper

from DataAugAndPreProc import PreprocDistBG
from DataAugAndPreProc import Preproc0
from DataAugAndPreProc import DataAug



@profile
def MemoryTester():
    # Rand Seed
    randSeed = 0

    #
    # Param Setting
    #
    #   Running Params
    classes = 7
    trainTestSplit = 0.8
    batchSizeTrain = 8
    slices = 3
    resize = (256, 256)  # None

    pathModel = "./model.pth"  # "./model_D2.pth"#"./model.pth"#"./model_6.pth"
    pathSrc = "../../../Sources/Data/data_nii"
    pathTarg = "../../../Sources/Data/output"  # "../../../Sources/Data/output_D2"#"../../../Sources/Data/output"#"../../../Sources/Data/output_6"
    pathRunningLossPlot = "../../../Sources/Data/loss/loss.jpg"

    aug = DataAug
    preproc = Preproc0

    np.random.seed(randSeed)

    pathSrcData = os.path.join(pathSrc, "data")
    pathSrcMask = os.path.join(pathSrc, "masks")

    niisData = NiiProcessor.ReadAllNiiInDir(pathSrcData)
    niisMask = NiiProcessor.ReadAllNiiInDir(pathSrcMask)

    # Split train set and test set
    niisAll = ImgDataSet.Split(niisData, niisMask, trainTestSplit)

    print("Making train set...")
    datasetTrain = ImgDataSet()
    datasetTrain.InitFromNiis(niisAll["niisDataTrain"], niisAll["niisMaskTrain"], slices=slices, classes=classes,
                              resize=resize, aug=aug, preproc=preproc)
    print("Making train loader...")
    loaderTrain = data.DataLoader(dataset=datasetTrain, batch_size=batchSizeTrain, shuffle=True)
    print("Done")

    print("Making test set...")
    datasetTest = ImgDataSet()
    datasetTest.InitFromNiis(niisAll["niisDataTest"], niisAll["niisMaskTest"], slices=slices, classes=classes,
                             resize=resize, aug=DataAug, preproc=preproc)
    print("Making test loader...")
    loaderTest = data.DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)
    print("Done")

if __name__ == '__main__':
    MemoryTester()