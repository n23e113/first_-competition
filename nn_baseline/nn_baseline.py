import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import getpass
import sys
import time
import numpy as np
from copy import deepcopy
import itertools
import tensorflow as tf

sys.path.append("..")
from q2_initialization import xavier_weight_init

tf.logging.set_verbosity(tf.logging.ERROR)

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

# for debug purpose
dumy_data_input_pos = np.random.uniform(0, 1, size=(4096, 100))
dumy_data_label_pos = np.ones(4096, dtype=np.int)
dumy_data_input_neg = np.random.uniform(-1, 0, size=(4096, 100))
dumy_data_label_neg = np.zeros(4096, dtype=np.int)
dumy_data_input = np.append(dumy_data_input_pos, dumy_data_input_neg, 0)
dumy_data_label = np.append(dumy_data_label_pos, dumy_data_label_neg, 0)

class Config(object):
  batch_size = 64
  input_dim = 100
  hidden_size_1 = 100
  hidden_size_2 = 100
  output_class = 2
  max_epochs = 10
  early_stopping = 2
  dropout = 0.5
  lr = 0.001
  l2 = 0.0

class NN_Baseline_Model():

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
    # for debug purpose
    self.train_data = {}
    self.valid_data = {}
    self.test_data = {}
    self.train_data['input'] = dumy_data_input
    self.train_data['label'] = dumy_data_label
    self.valid_data['input'] = dumy_data_input
    self.valid_data['label'] = dumy_data_label
    self.test_data['input'] = dumy_data_input
    self.test_data['label'] = dumy_data_label
    #print self.train_data['input']
    #print self.train_data['label']

  def setup_placeholders(self):
    self.input_placeholder = tf.placeholder(
      tf.float32, shape=(None, self.config.input_dim), name='NN_Baseline_Input')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='NN_Baseline_Dropout')
    self.label_placeholder = tf.placeholder(tf.int32, shape=(None), name='NN_Baseline_Label')

  def build_loss_op(self, logits, labels):
    #print 'logits.shape', logits.get_shape()
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #print 'reg_losses.shape', reg_losses.get_shape()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='NN_Baseline_CE_Per_Example')
    #print 'loss.shape', loss.get_shape()
    loss = tf.reduce_sum(loss, name='NN_Baseline_CE')
    #print 'reduce sum loss.shape', loss.get_shape()
    total_loss = reg_losses + loss
    #print 'total_loss.shape', total_loss.get_shape()
    #raw_input()
    train_predict = tf.argmax(tf.exp(logits), 1)
    return total_loss, train_predict

  def build_prediction(self, logits):
    return tf.argmax(tf.exp(logits), 1)

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
        tf.constant_initializer(0.0)
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
        tf.constant_initializer(0.0)
      )

      output_weights = tf.get_variable(
        'Output_Layer',
        [self.config.hidden_size_2, self.config.output_class],
        tf.float32,
        xavier_weight_init(),
        tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      output_bias = tf.get_variable(
        'Bias_output',
        [self.config.output_class],
        tf.float32,
        tf.constant_initializer(0.0)
      )

    local1 = tf.nn.relu(tf.matmul(tf.nn.dropout(inputs, dropout), h1) + b1)
    local2 = tf.nn.relu(tf.matmul(tf.nn.dropout(local1, dropout), h2) + b2)
    linear_output = tf.matmul(tf.nn.dropout(local2, dropout), output_weights) + output_bias

    return linear_output

  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.setup_placeholders()
    self.linear_output = self.build_model(self.input_placeholder, self.dropout_placeholder)
    self.calculate_loss, self.train_predict = self.build_loss_op(self.linear_output, self.label_placeholder)
    self.predict_op = self.build_prediction(self.linear_output)
    self.train_op = self.build_training_op(self.calculate_loss)


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    total_loss = []
    predict_result = None
    predict_op = self.train_predict
    if not train_op:
      train_op = tf.no_op()
      predict_op = self.predict_op
      dp = 1.0
    #total_steps = data['input'].shape[0] / self.config.batch_size
    for step in xrange(0, data['input'].shape[0], self.config.batch_size):
      #begin = step
      #end = step + self.config.batch_size
      #print 'begin, end', begin, end
      #raw_input()
      x = data['input'][step : step + self.config.batch_size]
      y = data['label'][step : step + self.config.batch_size]
      #print '*********'
      #print x
      #print y
      #print '---------'
      #print 'x.shape, y.shape', x.shape, y.shape
      feed = {self.input_placeholder: x,
              self.label_placeholder: y,
              self.dropout_placeholder: dp}
      loss, predict, _ = session.run(
        [self.calculate_loss, predict_op, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
          step, data['input'].shape[0], np.mean(total_loss)))
        sys.stdout.flush()
      if predict != None:
        if predict_result == None:
          predict_result = predict
          #print 'predict.shape', predict.shape
          #raw_input()
        else:
          predict_result = np.append(predict_result, predict)
          #print 'predict_result.shape', predict_result.shape
          #raw_input()
    if verbose:
      sys.stdout.write('\r\n')
    return np.mean(total_loss), predict_result

def make_conf(labels, predictions):
  #print labels
  #print predictions
  confmat = np.zeros([2, 2])
  for l,p in itertools.izip(labels, predictions):
    confmat[l, p] += 1
  return confmat

if __name__ == "__main__":
  config = Config()
  model = NN_Baseline_Model(config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config) as session:
    best_val = float('inf')
    best_val_epoch = 0
    save_path = None
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()

      train_loss, prediction = model.run_epoch(
          session, model.train_data, train_op=model.train_op)
      print 'Training loss: {}'.format(train_loss)
      print make_conf(model.train_data['label'], prediction)
      valid_loss, prediction = model.run_epoch(session, model.valid_data)
      print 'Validation loss: {}'.format(valid_loss)
      print make_conf(model.valid_data['label'], prediction)
      if valid_loss < best_val:
        best_val = valid_loss
        best_val_epoch = epoch
        save_path = saver.save(session, './nn_baseline.weights')
        print('Model saved in file: %s' % save_path)
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)

    saver.restore(session, save_path)
    test_loss, test_predict = model.run_epoch(session, model.test_data)
    print 'Test loss: {}'.format(test_loss)
    print make_conf(model.test_data['label'], test_predict)
