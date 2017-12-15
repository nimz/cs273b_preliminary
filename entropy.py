import numpy as np
import random
from sklearn import linear_model
import sklearn.metrics
import matplotlib.pyplot as plt

# input is numpy array of size batchSize * (2k_0 + 1) * 4, where k_0<k is a smaller window around the gene
# output is a numpy array of size batchSize * 1, where each location contains the entropy of that sequence
def entropySequence(sequenceStrings):
    lenSequenceString = sequenceStrings.shape
    entropyString = np.zeros(lenSequenceString[0])
    for j in range(4):
        probLetter = np.sum(sequenceStrings[:, :, j], axis = 1)/lenSequenceString[1]
        logProbLetter = np.log(probLetter)
        logProbLetter[logProbLetter == -np.Inf] = 0
        entropyString -= np.multiply(probLetter, logProbLetter)
    return entropyString


# input is numpy array of size batchSize * (2k + 1) * 4, ie. the entire batched dataset
# output is a numpy array of size batchSize * k
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
def entropyVector(sequenceStrings):
    lenSequenceString = sequenceStrings.shape
    centr = int((lenSequenceString[1]-1)/2)
    entropyVal = np.zeros((lenSequenceString[0], centr))
    for j in range(centr + 1, lenSequenceString[1]):
        entropyVal[:, j - centr - 1] = entropySequence(sequenceStrings[:, range((2*centr-j), (j+1)), :])
    return entropyVal


# input is numpy array of size batchSize * k (entropyStrings), numpy array of size batchSize * 1 (labels), trainIndex, testIndex
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
# prints metrics for logistic regression
def logisticRegression(entropyStrings, labels, trainIndex, testIndex):
    logReg = linear_model.LogisticRegression(C=1e5)
    logReg.fit(entropyStrings[trainIndex, :], labels[trainIndex])
    logRegPred = logReg.predict(entropyStrings[testIndex, :])
    print "Accuracy Score: %f" % sklearn.metrics.accuracy_score(labels[testIndex], logRegPred)
    print "F1 Score: %f" % sklearn.metrics.f1_score(labels[testIndex], logRegPred)
    print "ROCAUC Score: %f" % sklearn.metrics.roc_auc_score(labels[testIndex], logRegPred)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels[testIndex], logRegPred)
    print "PRAUC Score: %f" % sklearn.metrics.average_precision_score(labels[testIndex], logRegPred)
    print "Confusion Matrix:"
    print sklearn.metrics.confusion_matrix(labels[testIndex], logRegPred)
    k = entropyStrings.shape[1]
    plt.plot(range(1, k+1), logReg.coef_[0, :])
    plt.xlabel('Window Size')
    plt.ylabel('Regression Coefficient')
    plt.title('Variation of Regression Coefficient with Window Size')
    plt.savefig('LogRegCoeff.png')
    plt.clf()
    plt.plot(range(1, k+1), np.fabs(logReg.coef_[0, :]))
    plt.xlabel('Window Size')
    plt.ylabel('Absolute Regression Coefficient')
    plt.title('Variation of Absolute Regression Coefficient with Window Size')
    plt.savefig('LogRegCoeffAbs.png')
    plt.clf()
    np.save("/datadrive/project_data/LogRegCoeff.npy", logReg.coef_[0, :])
