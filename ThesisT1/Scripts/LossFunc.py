
import numpy as np
import torch.nn as tnn
from Utils import CommonUtil

#MACRO
DEBUG = False

class DiceLoss(tnn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    # input: Tensor[batch, class]
    def forward(self, output, target):
        if DEBUG:
            if np.any(np.isnan(output.cpu().detach().numpy())):
                raise Exception("NAN Warning!")
            if np.any(np.isnan(target.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        N = target.size(0)
        smooth = 1

        output_flat = output.view(N, -1)
        if DEBUG:
            if np.any(np.isnan(output_flat.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        target_flat = target.view(N, -1)
        if DEBUG:
            if np.any(np.isnan(target_flat.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        intersection = output_flat * target_flat
        if DEBUG:
            if np.any(np.isnan(intersection.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        inter_sum = intersection.sum(1)
        if DEBUG:
            print("inter_sum: ",inter_sum)
            if np.any(np.isnan(inter_sum.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        output_sum = output_flat.sum(1)
        if DEBUG:
            print("output_sum: ", output_sum)
            if np.any(np.isnan(output_sum.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        target_sum = target_flat.sum(1)
        if DEBUG:
            print("target_sum: ", target_sum)
            if np.any(np.isnan(target_sum.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        loss = 2 * (inter_sum + smooth) / (output_sum + target_sum + smooth)
        if DEBUG:
            if np.any(np.isnan(loss.cpu().detach().numpy())):
                raise Exception("NAN Warning!")

        loss = 1 - loss.sum() / N
        return loss


class MulticlassDiceLoss(tnn.Module):
    """	requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes	"""

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    # input: Tensor[batch, class, data(flatten)]
    # if no weight, loss is added together without scaling
    def forward(self, output, target, weights=None):
        # print("forward output:", output.shape)
        # print(output)
        # CommonUtil.MkFile("Test","output.npy")
        # np.save("Test/output.npy",output.cpu().detach().numpy())
        # print("forward target:",target.shape)
        # print(target)
        # CommonUtil.MkFile("Test", "target.npy")
        # np.save("Test/target.npy",target.cpu().detach().numpy())

        # input()
        # print("forward output:", np.any(np.isnan(output.cpu().detach().numpy())))
        # print("forward target:", np.any(np.isnan(target.cpu().detach().numpy())))


        C = target.shape[1]
        dice = DiceLoss()
        totalLoss = 0
        for i in range(C):
            diceLoss = dice(output[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            # print("diceLoss ",i," : ",diceLoss)
            totalLoss += diceLoss

        return totalLoss


def Test():
    pass

if __name__ == '__main__':
    Test()