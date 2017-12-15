import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import random
from sys import argv
import utils
import cs273b

data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

bsize = 5000

num_indels = []
seq = []
for i in range(22,23):#(1, 24):
  if i == 23:
    ch = 'X'
  else:
    ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr)
  
  insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
  deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
  indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1
  num_buckets = c_len//bsize
  num_indels_ch = [0]*num_buckets
  for il in indelLocations:
    if il//bsize >= len(num_indels_ch): break
    num_indels_ch[il // bsize] += 1
  num_indels.extend(num_indels_ch)
  del num_indels_ch, insertionLocations, deletionLocations, indelLocations
  seq_ch = np.array_split(referenceChr[:num_buckets*bsize], num_buckets)
  seq.extend(seq_ch)

del reference, referenceChr, seq_ch

from random import shuffle

order = [x for x in range(0, len(seq))]
shuffle(order)
seq = np.array([seq[i] for i in order])
num_indels = np.array([num_indels[i] for i in order])

ntest = len(seq) // 6
x_train = seq[:ntest]
y_train = num_indels[:ntest]
x_test = seq[ntest:]
y_test = num_indels[ntest:]

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

model = Sequential()
model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=(bsize, 4)))#, kernel_regularizer=l2(0.0001)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))#, kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='relu'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mae'])

batch_size = 5
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=6,
          verbose=1)
          #validation_data=(x_test, y_test))

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

ytestout = utils.flatten(model.predict(x_test, batch_size=5, verbose=1))
print('')
#print(list(zip(y_test, ytestout)))

from scipy import stats
from sklearn import linear_model

r, p = stats.pearsonr(y_test, ytestout)
print(r)
print(p)

plt.scatter(y_test, ytestout)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-10}$)')
plt.plot(y_test, y_test, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred.png')
