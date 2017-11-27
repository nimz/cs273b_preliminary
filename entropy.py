import numpy as np
import random
import indel_model
import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils
from sklearn import linear_model
from sklearn.metrics import accuracy_score

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 50
    strlen = 2*window+1
    batch_size = 200
    test_batch_size = 200
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 2
    print_every = 100 # print accuracy every 100 steps


# input is numpy array of size batchSize * (2k_0 + 1) * 4, where k_0<k is a smaller window around the gene
# output is a numpy array of size batchSize * 1, where each location contains the entropy of that sequence
def entropySequence(sequenceStrings):
	lenSequenceString = sequenceStrings.shape
	entropyString = np.zeros(lenSequenceString[0])
	for i in range(lenSequenceString[0]):
		for j in range(4):
			probLetter = sum(sequenceStrings[i, :, j])/lenSequenceString[1]
			if(probLetter > 0):
				entropyString[i] -= probLetter*np.log(probLetter)
	return entropyString


# input is numpy array of size batchSize * (2k + 1) * 4, ie. the entire batched dataset
# output is a numpy array of size batchSize * k
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
def entropyVector(sequenceStrings):
	lenSequenceString = sequenceStrings.shape
	centr = int((lenSequenceString[1]-1)/2)
	entropyVal = np.zeros((lenSequenceString[0], centr+1))
	for j in range(centr + 1, lenSequenceString[1]):
		entropyVal[:, j - centr - 1] = entropySequence(sequenceStrings[:, range((2*centr-j), (j+1)), :])
	return entropyVal


# input is numpy array of size batchSize * k (entropyStrings), numpy array of size batchSize * 1 (labels), trainIndex, testIndex
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
# output is training accuracy- can be modified at a later stage
def logisticRegression(entropyStrings, labels, trainIndex, testIndex):
	logReg = linear_model.LogisticRegression(C=1e5)
	logReg.fit(entropyStrings[trainIndex, :], labels[trainIndex])
	logRegPred = logReg.predict(entropyStrings[testIndex, :])
	return accuracy_score(labels[testIndex], logRegPred)


config = Config()
loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.05)


datset = loader.dataset
labls = utils.flatten(loader.labels)
print(labls.shape)
print(logisticRegression(entropyVector(datset), labls, range(loader.num_train_examples+1), range(loader.num_train_examples+1, loader.dataset.shape[0])))
