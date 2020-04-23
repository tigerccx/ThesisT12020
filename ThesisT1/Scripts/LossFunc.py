
import numpy as np
import torch.nn as tnn
from Utils import CommonUtil

#MACRO
DEBUG = False

class DiceLoss(tnn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    # Input: Tensor[batch, class]
    #        Tensor[batch, class]
    # Output: float
    def forward(self, output, target):
        # if DEBUG:
        #     if np.any(np.isnan(output.cpu().detach().numpy())):
        #         raise Exception("NAN Warning!")
        #     if np.any(np.isnan(target.cpu().detach().numpy())):
        #         raise Exception("NAN Warning!")

        N = target.size(0)
        smooth = 1

        output_flat = output.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = output_flat * target_flat
        inter_sum = intersection.sum(1)
        output_sum = output_flat.sum(1)
        target_sum = target_flat.sum(1)
        loss = (2 * inter_sum + smooth) / (output_sum + target_sum + smooth)

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