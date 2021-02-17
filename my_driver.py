import numpy as np
import util, datasets, binary, dumbClassifiers
import runClassifier, knn

if __name__ == "__main__":
    eps_curve_6 = runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6}), datasets.DigitData)
    print("result: ", eps_curve_6)
    runClassifier.plotCurve('epsilon-balls_6', eps_curve_6)
    pass
    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
