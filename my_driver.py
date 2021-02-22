import numpy as np
import util, datasets, binary, dumbClassifiers, os
import runClassifier, knn, perceptron
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # eps_curve_6 = runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6}), datasets.DigitData)
    # print("result: ", eps_curve_6)
    # runClassifier.plotCurve('epsilon-balls_6', eps_curve_6)


    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
    # runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
    # runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)

    # runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
    # h = perceptron.Perceptron({'numEpoch': 200})
    # h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
    # runClassifier.plotClassifier(np.array([7.3, 18.9]), 0.0)
    # runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)

    # generate curve for wu6 b
    num_epochs = 10
    train_acc_list = []
    test_acc_list = []
    for i in range(num_epochs):
        train_acc, test_acc, _= runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': i+1}), datasets.SentimentData)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    X = np.arange(num_epochs) + 1
    # runClassifier.plotCurve('Perceptron Learning Curve for 5 epochs', percetron_curve)
    plt.plot(X,train_acc_list, label="Training Accuracy")
    plt.plot(X,test_acc_list, label="Testing Accuracy")
    plt.title("Train/Test Accuracy VS Number of Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("graph_output","wu6_b" +".png"))
    pass