import torch
from torch import tensor
import torch.nn as tnn
import torch.nn.functional as tfunc

#
# First UNet
#
# Original Network
# file:///D:/Sources/Study/Thesis%20Study/Prof%20E%20Meijering/Deep%20Learning%20for%20Muscle%20Segmentation%20in%20MRI%20and%20DTI%20Images/SEMANTIC%20SEGMENTATION%20OF%20THIGH%20MUSCLE%20USING%202.5D.pdf
class UNet_0(tnn.Module):
    # Network Definition
    def __init__(self, sclices, classes):
        super(UNet_0, self).__init__()
        # Conv(↓)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = tnn.Conv2d(sclices, 32, 3, padding=1)
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
        # self.anisoUpconv1 = tnn.ConvTranspose2d(32, classes, (1, 11), padding=1)

        # Softmax
        self.softmax2d = tnn.Softmax2d()

    def forward_0(self, input):
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

#
# UNet from essay
#
class Con(tnn.Module):
    def __init__(self, nIn, n, k):
        super(Con, self).__init__()

        padding = int(k/2)
        self.conv1 = tnn.Conv2d(nIn,n,k,padding=padding)
        self.bn1 = tnn.BatchNorm2d(n)

    def forward(self, x):
        return tfunc.leaky_relu(self.bn1(self.conv1(x)))

class BlockUNet(tnn.Module):
    def __init__(self, nIn, ni):
        super(BlockUNet, self).__init__()

        self.con1 = Con(nIn, ni, 3)
        self.con2 = Con(ni, ni, 3)

    def forward(self, x):
        return self.con2(self.con1(x))

class EncUNet(tnn.Module):
    def __init__(self, nIn, n, L, dropoutRate=0.5, useMaxPool=True):
        super(EncUNet, self).__init__()

        ni = 2**L*n
        self.useMaxPool = useMaxPool
        if useMaxPool:
            self.maxpool1 = tnn.MaxPool2d(2)
        self.block1 = BlockUNet(nIn,ni)
        self.dropout1 = tnn.Dropout2d(dropoutRate,inplace=True)

    def forward(self, x):
        if self.useMaxPool:
            return self.dropout1(self.block1(self.maxpool1(x)))
        else:
            return self.dropout1(self.block1(x))

class DecUNet(tnn.Module):
    def __init__(self, nIn, n, L, hOrgIn=None, wOrgIn=None, dropoutRate=0.5):
        if (hOrgIn is None and wOrgIn is not None) or (hOrgIn is not None and wOrgIn is None):
            raise Exception("Both xOrgIn and yOrgIn must be assigned together. ")
        super(DecUNet, self).__init__()

        ni = 2**L*n
        if hOrgIn is None:
            self.upsample = tnn.Upsample(scale_factor=2)
        else:
            self.upsample = tnn.Upsample(size=(nIn,int(hOrgIn/(2**L)),int(wOrgIn/(2**L))))
        self.block1 = BlockUNet(nIn, ni)
        self.dropout1 = tnn.Dropout2d(dropoutRate, inplace=True)

    def forward(self, x, xConcat):
        return self.dropout1(self.block1(torch.cat([xConcat,self.upsample(x)],dim=1)))

class UNet_1(tnn.Module):
    def __init__(self, n, slices, classes, depth=5, inputHW=None, dropoutRate=0.5):
        super(UNet_1,self).__init__()

        self.n = n
        self.slices = slices
        self.classes = classes
        self.depth = depth
        self.dropoutRate = dropoutRate

        self.con1=Con(slices, n, 1)
        self.enc0=EncUNet(n,n,0,dropoutRate=dropoutRate,useMaxPool=False)
        for l in range(1,depth):
            setattr(self,"enc"+str(l),EncUNet(2**(l-1)*n,n,l,dropoutRate=dropoutRate))
        if inputHW is None:
            for l in range(depth-2,-1,-1):
                setattr(self,"dec"+str(l), DecUNet(2**(l+1)*n+2**l*n,n,l,dropoutRate=dropoutRate))
        else:
            self.inputHW = inputHW
            for l in range(depth-2,-1,-1):
                setattr(self,"dec"+str(l), DecUNet(2**(l+1)*n+2**l*n,n,l,hOrgIn=inputHW[0],wOrgIn=inputHW[1],dropoutRate=dropoutRate))
        self.con2=Con(n, classes, 1)

    def forward(self, x):
        self.x1 = self.con1(x)
        self.xEnc0 = self.enc0(self.x1)
        strExec = ""
        for l in range(1,self.depth):
            strExec+="self.xEnc"+str(l)+"=self.enc"+str(l)+"(self.xEnc"+str(l-1)+");"
        exec(strExec)
        firstDecL = self.depth-2
        strExec = "self.xDec"+str(firstDecL)+"=self.dec"+str(firstDecL)+"(self.xEnc"+str(firstDecL+1)+",self.xEnc"+str(firstDecL)+");"
        for l in range(self.depth-3,-1,-1):
            strExec+="self.xDec"+str(l)+"=self.dec"+str(l)+"(self.xDec"+str(l+1)+",self.xEnc"+str(l)+");"
        exec(strExec)
        self.x2 = self.con2(self.xDec0)
        return self.x2