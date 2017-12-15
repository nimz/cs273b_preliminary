import cs273b
import load_coverage as lc
import load_recombination as lr
import numpy as np
from math import ceil
import utils
import entropy

data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
	def __init__(self, _kw=0, chromosome=21, windowSize=100, batchSize=100, testBatchSize=500, seed=1, test_frac=0.05, pos_frac=0.5, load_coverage=True, load_entropy=False, load_recombination=False, include_filtered=True, triclass=False, nearby=0, offset=0, load_entire = True, delref=True):
		self.window = windowSize
		self.batchSize = batchSize
		self.testBatchSize = testBatchSize
		self.test_frac = test_frac
		self.triclass = triclass
		self.nearby = nearby
		self.offset = offset
		self.load_entropy = load_entropy
                self.load_coverage = load_coverage
                self.load_recombination = load_recombination
		reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
		self.referenceChr = reference[str(chromosome)]
		self.refChrLen = len(self.referenceChr)
		del reference, ambiguous_bases
		ext = ".txt"
		if not include_filtered: ext = "_filtered" + ext
		if self.triclass:
			self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
			self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
			self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations))
		else:
			self.indelLocations = np.loadtxt(data_dir + "indelLocations{}".format(chromosome) + ext).astype(int)
		self.nonzeroLocationsRef = np.where(np.any(self.referenceChr != 0, axis = 1))[0]
		if nearby:
		  self.zeroLocationsRef = np.where(np.all(self.referenceChr == 0, axis = 1))[0]
		  self.setOfZeroLocations = set(self.zeroLocationsRef)
		self.indelLocations = self.indelLocations - offset
		self.coverage = None
		if load_coverage:
			self.coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(chromosome))
		self.recombination = None
                if load_recombination:
                	self.recombination = lr.load_recombination(data_dir + "recombination_map/genetic_map_chr{}_combined_b37.txt".format(chromosome))
                self.setOfIndelLocations = set(self.indelLocations)
		self.prevChosenRefLocations = set()
		self.cur_index = 0
		self.test_index = 0
		self.chrom_index = 0
		if seed is not None:
			np.random.seed(seed)
		self.__initializeTrainData(pos_frac)
		self.__initializeTestData()
		if not load_entire:
			del self.referenceChr
			del self.nonzeroLocationsRef

	# Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
	# Also includes desired number of negative training examples (positions not listed as indels)
	def __initializeTrainData(self, frac_positives):
		k = self.window # for brevity
		lengthIndels = len(self.indelLocations)
		num_negatives = int((1./frac_positives-1) * lengthIndels)
		total_length = lengthIndels + num_negatives
		dataset = np.zeros((total_length, 2*k + 1, 4))
		coverageDataset = np.zeros((total_length, 2*k + 1))
		entropyDataset = np.zeros((total_length, 2*k + 1))
                recombinationDataset = np.zeros((total_length, 1))
	        #recombinationDataset= np.zeros((total_length, 2*k + 1))
		if self.triclass:
		  labeltype = np.uint8
		else:
		  labeltype = np.bool
		labels = np.zeros(total_length, dtype=labeltype)
		genome_positions = np.zeros(total_length, dtype=np.uint32)
		num_negatives = int((1./frac_positives-1) * lengthIndels)

		# dataset should have all the indels as well as random negative training samples
		if self.nearby:
		  neg_positions = np.random.choice(self.indelLocations, size=num_negatives)
		  self.nearby_indels = neg_positions
		  offset = np.multiply(np.random.randint(1, self.nearby+1, size=num_negatives), np.random.choice([-1, 1], size=num_negatives))
		  neg_positions = neg_positions + offset # locations that are offset from indels by some amount
		  self.indices = neg_positions
		else:
		  neg_positions = np.random.choice(self.nonzeroLocationsRef, size=num_negatives)
		  self.indices = neg_positions
		  self.nearby_indels = neg_positions # to prevent error if this is undefined
		for i in range(lengthIndels + num_negatives):
			if i < lengthIndels:
				if not self.triclass:
				  label = 1 # standard binary classification labels
				elif i < len(self.insertionLocations):
				  label = 1 # insertions will be labeled as 1
				else:
				  label = 2 # deletions will be labeled as 2
				pos = self.indelLocations[i]
			else:
				label = 0
				pos = neg_positions[i - lengthIndels]
				if self.nearby:
				  niter = 0
				  while (pos in self.prevChosenRefLocations) or (pos in self.setOfZeroLocations) or (pos in self.setOfIndelLocations) and niter < 1001:
					self.nearby_indels[i - lengthIndels] = np.random.choice(self.indelLocations)
					pos = self.nearby_indels[i - lengthIndels] + np.random.randint(1, self.nearby+1) * np.random.choice([-1, 1])
					niter += 1
				else:
				  while (pos in self.prevChosenRefLocations) or (pos in self.setOfIndelLocations):
					pos = np.random.choice(self.nonzeroLocationsRef)
				self.indices[i - lengthIndels] = pos
				self.prevChosenRefLocations.add(pos)
			coverageWindow = np.zeros(2*k + 1)
			# get k base pairs before and after the position
			window = self.referenceChr[pos - k : pos + k + 1]
			coverageWindow = None
			if self.coverage is not None:
				coverageWindow = utils.flatten(self.coverage[pos - k : pos + k + 1])
			recombWindowAverage = None
                        if self.recombination is not None:
                                recombWindow = np.zeros((2*k + 1, 1))
                                recombWindowIndices = np.arange(pos - k, pos + k + 1).reshape((2*k + 1, 1))
                                recombInBounds = recombWindowIndices[np.where(recombWindowIndices < len(self.recombination))]
                                recombWindow[recombInBounds - (pos - k)] = self.recombination[recombInBounds]
                                recombOutOfBounds = recombWindowIndices[np.where(recombWindowIndices >= len(self.recombination))]
                                recombWindow[recombOutOfBounds - (pos - k)] = self.recombination[-1] 
                        	recombWindowAverage = np.mean(recombWindow)
				#recombWindowAverage = utils.flatten(recombWindow)
                        dataset[i] = window
			coverageDataset[i] = coverageWindow
                        recombinationDataset[i] = recombWindowAverage
			labels[i] = label
			genome_positions[i] = pos
		self.indices = np.concatenate((self.indelLocations, self.indices))
		self.nearby_indels = np.concatenate((self.indelLocations, self.nearby_indels))
		if self.load_entropy:
			entropyDataset[:, k+1:2*k+1] = entropy.entropyVector(dataset)
		rawZipped = zip(list(dataset), list(coverageDataset), list(labels), list(genome_positions), list(self.indices), list(self.nearby_indels), list(entropyDataset), list(recombinationDataset))
		# Shuffle the list
		np.random.shuffle(rawZipped)
		a, b, c, d, e, f, g, h = zip(*rawZipped)
		dataset = np.array(a)
		coverageDataset = np.array(b)
		entropyDataset = np.array(g)
                recombinationDataset = np.array(h)
		labels = np.array(c, dtype=labeltype)
		genome_positions = np.array(d, dtype=np.uint32)
		self.indices = np.array(e, dtype=np.uint32)
		self.nearby_indels = np.array(f, dtype=np.uint32)
		self.dataset = dataset
		self.coverageDataset = coverageDataset
		self.entropyDataset = entropyDataset
                self.recombinationDataset = recombinationDataset
		if self.triclass:
		  self.labels = utils.to_onehot(labels, 3)
		else:
		  self.labels = np.expand_dims(labels, axis=1)
		self.genome_positions = genome_positions
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
                retval = [self.dataset[batch_indices]]
                if self.load_coverage:
                	retval.append(self.coverageDataset[batch_indices])
                if self.load_entropy:
			retval.append(self.entropyDataset[batch_indices])
                if self.load_recombination:
			retval.append(self.recombinationDataset[batch_indices])
		retval.append(self.labels[batch_indices])

		#if self.coverage is not None and self.load_entropy:
		#	retval = (self.dataset[batch_indices], self.coverageDataset[batch_indices], self.entropyDataset[batch_indices], self.labels[batch_indices])
		#elif self.coverage is not None:
		#	retval = (self.dataset[batch_indices], self.coverageDataset[batch_indices], self.labels[batch_indices])
		#elif self.load_entropy:
		#	retval = (self.dataset[batch_indices], self.entropyDataset[batch_indices], self.labels[batch_indices])
		#else:
		#	retval = (self.dataset[batch_indices], self.labels[batch_indices])
		self.cur_index = end
		if end >= self.num_train_examples:
			self.cur_index = 0
		#return retval
		return tuple(retval)

	def __initializeTestData(self):
		# Get all non-training examples
		test_data_x = self.dataset[self.num_train_examples+1:]
		test_data_y = self.labels[self.num_train_examples+1:]
                self.test_data = [test_data_x]
                if self.load_coverage:
			self.test_data.append(self.coverageDataset[self.num_train_examples+1:])
		if self.load_entropy:
			self.test_data.append(self.entropyDataset[self.num_train_examples+1:])
		if self.load_recombination:
			self.test_data.append(self.recombinationDataset[self.num_train_examples+1:])
		self.test_data.append(test_data_y)
		self.test_data = tuple(self.test_data)
		#if self.coverage is not None and self.load_entropy:
		#	test_data_coverage = self.coverageDataset[self.num_train_examples+1:]
		#	test_data_entropy = self.entropyDataset[self.num_train_examples+1:]
		#	self.test_data = test_data_x, test_data_coverage, test_data_entropy, test_data_y
		#elif self.coverage is not None:
		#	test_data_coverage = self.coverageDataset[self.num_train_examples+1:]
		#	self.test_data = test_data_x, test_data_coverage, test_data_y
		#elif self.load_entropy:
		#	test_data_entropy = self.entropyDataset[self.num_train_examples+1:]
		#	self.test_data = test_data_x, test_data_entropy, test_data_y
		#else:
		#	self.test_data = test_data_x, test_data_y
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
			rv = [self.test_data[i][self.test_index:self.test_index+self.testBatchSize] for i in range(len(self.test_data))]
			rv = tuple(rv)
			#if self.coverage is not None and self.load_entropy:
			#	rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[2][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[3][self.test_index:self.test_index+self.testBatchSize]
			#elif self.coverage is not None:
			#	rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[2][self.test_index:self.test_index+self.testBatchSize]
			#elif self.load_entropy:
			#	rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[2][self.test_index:self.test_index+self.testBatchSize]
			#else:
			#	rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
			#		 self.test_data[1][self.test_index:self.test_index+self.testBatchSize]
		else:
			raise RuntimeError("test index is {}, only {} examples".format(self.test_index, len(self.test_data[1])))
		self.test_index += self.testBatchSize
		return rv

	def val_set(self, length=1000):
		retval = [self.test_data[i][:length] for i in range(len(self.test_data))]
		return tuple(retval)
		#if self.coverage is not None and self.load_entropy:
		#	return self.test_data[0][:length], self.test_data[1][:length], self.test_data[2][:length], self.test_data[3][:length]
		#elif self.coverage is not None:
		#	return self.test_data[0][:length], self.test_data[1][:length], self.test_data[2][:length]
		#elif self.load_entropy:
		#	return self.test_data[0][:length], self.test_data[1][:length], self.test_data[2][:length]
		#else:
		#	return self.test_data[0][:length], self.test_data[1][:length]

	def load_chromosome_window_batch(self, window_size, batch_size):
		lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
		ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
		num_ex = ub - lb
		X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
		self.chrom_index = ub
		labels = [ex in self.setOfIndelLocations for ex in range(lb, ub)]
		return self.referenceChr[X+Y, :], labels, lb, ub
