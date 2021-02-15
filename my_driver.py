import numpy as np
import util, datasets, binary, dumbClassifiers
import runClassifier, knn

if __name__ == "__main__":
    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)