import tensorflow as tf
import numpy as np

## DNA Processing
letters = ['A', 'C', 'G', 'T']

def to_letter(lst):
  x = next((i for i, x in enumerate(lst) if x), None)
  if x is not None: return letters[x]
  return '?'

def onehot_to_str(elem):
  elem = list(elem)
  ret = []
  for l in elem:
    ret.append(to_letter(list(l)))
  return ''.join(ret)

def batch_to_strs(batch_x):
  b = list(batch_x)
  ret = []
  for elem in b:
    ret.append(onehot_to_str(elem))
  return ret

def flatten(arr):
  return np.reshape(arr, -1)

def to_onehot(labels, num_categories):
  rv = np.zeros((len(labels), num_categories), dtype=np.uint8)
  for i, label in enumerate(labels):
    rv[i][label] = 1
  return rv

## Tensor flow helper methods
def weight_variable(shape):
  # Xavier initialization
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))

def weight_variable_reg(name, shape, beta):
  initializer = tf.contrib.layers.xavier_initializer()
  if beta:
    return tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer = tf.contrib.layers.l2_regularizer(beta))
  return tf.get_variable(name=name, shape=shape, initializer=initializer)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 1D convolution with stride 1 and zero padding
def conv1d(x, W):
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

# Cross entropy loss. Higher weight_falsepos will penalize false positives more relative to false negatives
def cross_entropy(y_pred, y_true, weight_falsepos=1):
  # Add eps to prevent errors in rare cases of 0 input to log
  eps = 1e-12
  return tf.reduce_mean(-y_true * tf.log(y_pred + eps) - weight_falsepos * (1-y_true) * tf.log(1-y_pred + eps))

# Leaky ReLU
def lrelu(x, alpha=0.01):
  return tf.maximum(x, alpha * x)

def adam_opt(loss, start_lr, decay_every_num_batches=0, decay_base=0.98):
  adam = tf.train.AdamOptimizer
  if not decay_every_num_batches:
    return adam(start_lr).minimize(loss)
  global_step = tf.Variable(0, trainable=False)
  lr = tf.train.exponential_decay(start_lr, global_step, decay_every_num_batches,
                                  decay_base, staircase=True)
  return adam(lr).minimize(loss, global_step=global_step)

def compute_accuracy(y_pred, y_true):
  if y_pred.get_shape().as_list()[-1] == 1:
    correct_prediction = tf.equal(tf.round(y_pred), y_true)
  else: # multiclass: we are dealing with logits, not probabilities
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y_true, axis=-1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

def coverage_placeholder(length):
  return tf.placeholder(tf.float32, shape=None)
  #return tf.placeholder(tf.float32, shape=[None, length])

def dna_placeholder(length):
  # 4 because the data is one-hot encoded
  return tf.placeholder(tf.float32, shape=[None, length, 4])
