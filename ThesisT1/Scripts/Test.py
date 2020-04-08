
import numpy as np
from matplotlib import pylab as plt


if __name__ == '__main__':
    pathRunningLossPlot = 'testOutput.jpg'
    runningLoss = [1,2,3,4,5]
    runningLossVali = [2,5,4,7,1]
    epochs = 5
    X = np.arange(1, epochs + 1, 1)
    Y = runningLoss
    Y1 = runningLossVali
    plt.title("Running Loss")
    plt.plot(X, Y, label="Training Loss")

    plt.plot(X, Y1, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(pathRunningLossPlot)
    plt.cla()