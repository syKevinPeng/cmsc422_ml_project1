from math import *
from imports import *
import random
from numpy import *
import matplotlib.pyplot as plt
import os
import KNNDigits


waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)

def computeDistances(data):
    N = len(data)
    D = len(data[0])
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append( computeExampleDistance(data[n],data[m])  / sqrt(D))
    return dist
if __name__ == "__main__":
    N    = 200                   # number of examples
    Dims = [784] #[2, 8, 32, 128, 512]   # dimensionalities to try 784
    Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
    Bins = arange(0, 1, 0.02)

    plt.xlabel('distance / sqrt(dimensionality)')
    plt.ylabel('# of pairs of points at that distance')
    plt.title('dimensionality versus uniform point distances')

    for i,d in enumerate(Dims):
        distances = computeDistances(datasets.loadDigitData('data/1vs2.all'))
        print ("D=%d, average distance=%g" % (d, mean(distances) * sqrt(d)))
        plt.hist(distances,
                 Bins,
                 histtype='step',
                 color=Cols[i])
        if waitForEnter:
            plt.legend(['%d dims' % d for d in Dims])
            plt.show(False)
            x = raw_input('Press enter to continue...')

    plt.legend(['%d dims' % d for d in Dims])
    plt.savefig(os.path.join("graph_output", "histogramA" + ".png"))
    plt.show()

