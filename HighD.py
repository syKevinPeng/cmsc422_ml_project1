from math import *
from imports import *
import random
from numpy import *
import matplotlib.pyplot as plt
import os

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2, dims):
    dist = 0.0
    for d in dims:
        dist += (x1[d-1] - x2[d-1]) * (x1[d-1] - x2[d-1])
    return sqrt(dist)

def computeDistancesSubdims(data, d):
    N = len(data[0])
    print(N)
    D = len(data[0][0])
    print(D)
    # select d from 784 diminsion
    print(d)
    dRange = range(1, 784)
    dims = np.random.choice(dRange, d)
    print(dims)

    #calculate distance according to dims selected
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append( computeExampleDistance(data[0][n], data[0][m], dims)  / sqrt(d))
    return dist
if __name__ == "__main__":
    N    = 200                   # number of examples
    Dims = [2, 8, 32, 128, 512]   # dimensionalities to try 784
    Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
    Bins = arange(0, 1, 0.02)

    plt.xlabel('distance / sqrt(dimensionality)')
    plt.ylabel('# of pairs of points at that distance')
    plt.title('dimensionality versus uniform point distances')

    for i,d in enumerate(Dims):
        distances = computeDistancesSubdims(datasets.loadDigitData('data/1vs2.all'), d)
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
    plt.savefig(os.path.join("graph_output", "histogramC" + ".png"))
    plt.show()

