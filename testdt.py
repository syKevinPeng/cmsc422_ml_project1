import dt
from imports import *

# simple depth 1 decision tree
print("Depth 1 dt:")
h = dt.DT({'maxDepth': 1})
print(h)

h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

# depth 2 decision tree
print("Depth 2 dt:")
h = dt.DT({'maxDepth': 2})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

# depth 5 decision tree
print("Depth 5 dt:")
h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

# depth 2 with sentimentdata decision tree
print("Depth 2 dt with sentiment data")
h = dt.DT({'maxDepth': 2})
h.train(datasets.SentimentData.X, datasets.SentimentData.Y)
print(h)

print('branch word:')
print(datasets.SentimentData.words[626])
print(datasets.SentimentData.words[683])
print(datasets.SentimentData.words[1139])



#train/test accuracy of decision tree
print("train/test accuracy:")
runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)