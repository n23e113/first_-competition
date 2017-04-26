import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import getpass
import sys
import time
import numpy as np
from copy import deepcopy
import tensorflow as tf

from q2_initialization import xavier_weight_init

tf.logging.set_verbosity(tf.logging.ERROR)

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  batch_size = 64
  iput_dim = 100
  hidden_size_1 = 100
  hidden_size_2 = 100
  output_class = 2
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.008
  l2 = 0.0

class NN_Baseline_Model(LanguageModel):

  def load_data(self, debug=False):
    '''
    load data from ndarray file

    self.train
    self.valid
    self.test
    if debug:
      num_debug = 1024
      self.train = self.train[:num_debug]
      self.valid = self.train[:num_debug]
      self.test =  = self.train[:num_debug]
    '''

  def setup_placeholders(self):
    self.input_placeholder = tf.placeholder(
      tf.int32, shape=(None, self.config.iput_dim), name='NN_Baseline_Input')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='NN_Baseline_Dropout')
    self.label_placeholder = tf.placeholder(tf.int32, shape=(None), name='NN_Baseline_Label')

  def build_loss_op(self, logits, labels):
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='NN_Baseline_CE_Per_Example')
    loss = tf.reduce_sum(loss, name='NN_Baseline_CE')
    total_loss = reg_losses + loss
    return total_loss

  def build_training_op(self, loss):
    train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
    return train_op

  def build_model(self, inputs, dropout):

    with tf.variable_scope('NN_Baseline') as scope:
      h1 = tf.get_variable(
        'Hidden_Layer_1',
        [self.config.input_dim, self.config.hidden_size_1],
        tf.float32,
        xavier_weight_init(),
        tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      b1 = tf.get_variable(
        'Bias_1',
        [self.config.hidden_size_1],
        tf.float32,
        tf.constant_initializer(0.1)
      )

      h2 = tf.get_variable(
        'Hidden_Layer_2',
        [self.config.hidden_size_1, self.config.hidden_size_2],
        tf.float32,
        xavier_weight_init(),
        tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      b2 = tf.get_variable(
        'Bias_2',
        [self.config.hidden_size_2],
        tf.float32,
        tf.constant_initializer(0.1)
      )

      output_weights = tf.get_variable(
        'Output_Layer',
        [self.config.hidden_size_2, output_class],
        tf.float32,
        xavier_weight_init(),
        tf.contrib,layers.l2_regularizer(self.cofig.l2)
      )

      output_bias = tf.get_variable(
        'Bias_output',
        [self.config.hidden_size_2],
        tf.float32,
        tf.constant_initializer(0.1)
      )

    local1 = tf.nn.relu(tf.matmul(tf.nn.dropout(input, dropout), h1) + b1)
    local2 = tf.nn.relu(tf.matmul(tf.nn.dropout(local1, dropout), h2) + b2)
    linear_output = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(local2, dropout), output_weights) + output_bias)

    return linear_output

  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.setup_placeholders()
    self.linear_output = self.build_model(self.input_placeholder, self.dropout_placeholder)
    self.calculate_loss = self.build_loss_op(linear_output, self.label_placeholder)
    self.train_op = self.build_training_op(self.calculate_loss)

  def run_epoch(self, session, data, train_op=None, verbose=100):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    for x, y in enum(data, self.config.batch_size):
      feed = {self.input_placeholder: x,
              self.label_placeholder: y,
              self.dropout_placeholder: dp}
      loss, _ = session.run(
          [self.calculate_loss, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.mean(total_loss)))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.mean(total_loss)

if __name__ == "__main__":
    print 'hello'
