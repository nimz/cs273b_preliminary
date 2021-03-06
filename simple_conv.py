import tensorflow as tf
import numpy as np
import random
from sys import argv
import indel_model
import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 40
    strlen = 2*window+1
    batch_size = 50
    test_batch_size = 200
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 5
    print_every = 100 # print accuracy every 100 steps

class SimpleConv(indel_model.IndelModel):
#    def add_placeholders(self):
#        self.x = utils.dna_placeholder(2*self.config.window+1)
#        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        fs = [5, 5] # filter sizes
        cs = [4, 40, 80] # cs[i] is output number of channels from layer i [where layer 0 is input layer]

        # First conv layer
        W_conv1 = utils.weight_variable([fs[0], cs[0], cs[1]])
        b_conv1 = utils.bias_variable([cs[1]])

        h_conv1 = utils.lrelu(utils.conv1d(self.x, W_conv1) + b_conv1)

        # Second conv layer
        W_conv2 = utils.weight_variable([fs[1], cs[1], cs[2]])
        b_conv2 = utils.bias_variable([cs[2]])

        h_conv2 = utils.lrelu(utils.conv1d(h_conv1, W_conv2) + b_conv2)

        # First fully connected layer. Reshape the convolution output to 1D vector
        W_fc1 = utils.weight_variable([self.config.strlen * cs[2], 1024])
        b_fc1 = utils.bias_variable([1024])

        h_conv2_flat = tf.reshape(h_conv2, [-1, self.config.strlen * cs[2]])
        h_fc1 = utils.lrelu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # Dropout (should be added to earlier layers too...)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Final fully-connected layer
        W_fc2 = utils.weight_variable([1024, 1])
        b_fc2 = utils.bias_variable([1])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_out = tf.sigmoid(y_conv)

        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

if len(argv) > 1:
  chromosome = int(argv[1])
else:
  chromosome=21
config = Config()
loader = load_dataset.DatasetLoader(chromosome=chromosome, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)

conv_net = SimpleConv(config, loader)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
'''losses, val_accuracies = conv_net.fit(sess, save=True)

conv_net.predictAll(sess, save=True)
print("test accuracy %g" % conv_net.test(sess))

all_results = conv_net.hard_examples(sess)
hard_positives = [x for x in all_results if x[1]]
#print(all_results[:100])
#print(hard_positives[:100])

auroc = conv_net.calc_roc(sess, 'conv_auroc.png')
print("ROC AUC: %g" % auroc)
auprc = conv_net.calc_auprc(sess, 'conv_auprc.png')
print("PR AUC: %g" % auprc)
print("f1 score: %g" % conv_net.calc_f1(sess))
conv_net.print_confusion_matrix(sess)

conv_net.plot_val_accuracies('conv_val.png')'''
conv_net.print_metrics(sess, 'conv', 'simple_conv_results.txt')
