import cs273b
import load_coverage as lc
import numpy as np
from math import ceil
import utils

data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
    def __init__(self, _kw=0, chromosome=21, windowSize=100, batchSize=100, testBatchSize=500, seed=1, test_frac=0.05, pos_frac=0.5, load_coverage=True):
        self.window = windowSize
        self.batchSize = batchSize
        self.testBatchSize = testBatchSize
        self.test_frac = test_frac
        reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
        self.referenceChr = reference[str(chromosome)]
        self.indelLocations = np.loadtxt(data_dir + "indelLocations{}.txt".format(chromosome)).astype(int)
        self.nonzeroLocationsRef = np.where(np.any(self.referenceChr != 0, axis = 1))[0]
        self.coverage = None
        if load_coverage:
          self.coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(chromosome))
        self.setOfIndelLocations = set(self.indelLocations)
        self.prevChosenRefLocations = set()
        self.cur_index = 0
        self.test_index = 0
        if seed is not None:
            np.random.seed(seed)
        self.__initializeTrainData(pos_frac)
        self.__initializeTestData()
        self.referenceChr = None # remove reference to preserve memory
        self.nonzeroLocationsRef = None
        self.coverage = None

    # Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
    # Also includes desired number of negative training examples (positions not listed as indels)
    def __initializeTrainData(self, frac_positives):
        k = self.window # for brevity
        lengthIndels = len(self.indelLocations)
        num_negatives = int((1./frac_positives-1) * lengthIndels)
        total_length = lengthIndels + num_negatives
        dataset = np.zeros((total_length, 2*k + 1, 4))
        coverageDataset = np.zeros((total_length, 2*k + 1))
        labels = np.zeros(total_length, dtype=np.int)
        num_negatives = int((1./frac_positives-1) * lengthIndels)
        # dataset should have all the indels as well as random negative training samples
        neg_positions = np.random.choice(self.nonzeroLocationsRef, size=num_negatives)
        for i in range(lengthIndels + num_negatives):
            if i < lengthIndels:
                label = 1
                pos = self.indelLocations[i]
            else:
                label = 0
                pos = neg_positions[i - lengthIndels]
                while (pos in self.prevChosenRefLocations) or (pos in self.setOfIndelLocations):
                    pos = np.random.choice(self.nonzeroLocationsRef)
                self.prevChosenRefLocations.add(pos)
            coverageWindow = np.zeros(2*k + 1)
            # get k base pairs before and after the position
            window = self.referenceChr[pos - k : pos + k + 1]
            coverageWindow = None
            if self.coverage is not None:
                coverageWindow = utils.flatten(self.coverage[pos - k : pos + k + 1])
            dataset[i] = window
            coverageDataset[i] = coverageWindow
            labels[i] = label
        rawZipped = zip(list(dataset), list(coverageDataset), list(labels))
        # Shuffle the list
        np.random.shuffle(rawZipped)
        a, b, c = zip(*rawZipped)
        dataset = np.array(a)
        coverageDataset = np.array(b)
        labels = np.array(c)
        self.dataset = dataset
        self.coverageDataset = coverageDataset
        self.labels = np.expand_dims(labels, axis=1)
        self.num_train_examples = int(round(total_length * (1-self.test_frac)))
        self.ordering = list(range(0, self.num_train_examples))

    def get_batch(self):
        return self.get_randbatch() # default: random

    def get_randbatch(self, batchSize=0):
        if batchSize == 0: batchSize = self.batchSize
        # Randomize the order of batches
        if self.cur_index == 0:
            np.random.shuffle(self.ordering)
        start, end = self.cur_index, self.cur_index + batchSize
        batch_indices = self.ordering[start : end]
        retval = (self.dataset[batch_indices], self.labels[batch_indices])
        self.cur_index = end
        if end >= self.num_train_examples:
            self.cur_index = 0
        return retval

    def __initializeTestData(self):
        # Get all non-training examples
        test_data_x = self.dataset[self.num_train_examples+1:]
        test_data_y = self.labels[self.num_train_examples+1:]
        self.test_data = test_data_x, test_data_y
        print("Number of test examples: {}".format(len(test_data_y)))

    def num_trainbatches(self):
        return int(ceil(float(self.num_train_examples) / self.batchSize))

    def len_testdata(self):
        return len(self.test_data[1])

    def num_testbatches(self):
        return int(ceil(float(len(self.test_data[1])) / self.testBatchSize))

    def reset_test_index(self):
        self.test_index = 0

    def get_testbatch(self):
        if self.test_index < len(self.test_data[1]):
            rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
                 self.test_data[1][self.test_index:self.test_index+self.testBatchSize]
        else:
            raise RuntimeError("test index is {}, only {} examples".format(self.test_index, len(self.test_data[1])))
        self.test_index += self.testBatchSize
        return rv

    def val_set(self, length=1000):
        return self.test_data[0][:length], self.test_data[1][:length]
