import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import os.path
import getpass
import sys
import time
import numpy as np
from copy import deepcopy
import itertools
import unicodecsv as csv
from concurrent.futures import ProcessPoolExecutor
import functools
import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training model", action="store_true")
parser.add_argument("--valid", help="test model use training data 7|1.5|1.5", action="store_true")
parser.add_argument("--test", help="test model use training data 7|1.5|1.5", action="store_true")
parser.add_argument("--infer", help="output predict", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5, help="predict threshold, if predict[0] > threshold : male")
parser.add_argument("--globaldebug", help="turn global debug on, use hardcoded testing data", action="store_true")
parser.add_argument("--undersample", help="use undersampled training data", action="store_true")
args = parser.parse_args()

import tensorflow as tf

sys.path.append("..")
from q2_initialization import xavier_weight_init

tf.logging.set_verbosity(tf.logging.ERROR)

Global_Debug = args.globaldebug
Undersample_Array = args.undersample
print 'Global_Debug : ', Global_Debug, ' \nUndersample_Array : ', Undersample_Array
print 'threshold : ', args.threshold
Undersample_Numpy_Array_File = '../training_data/undersample_numpy_array.npy'
Undersample_Numpy_Array_Files = '../training_data/undersample_numpy_array_'
#Oversample_Numpy_Array_File = '/media/lyk/317094AE34F3BB64/first_competition_npy/training_data/training_data_full/oversample_numpy_array.npy'
#Oversample_Numpy_Array_Files = '/media/lyk/317094AE34F3BB64/first_competition_npy/training_data/training_data_full/oversample_numpy_array_'
#Oversample_Numpy_Array_File = '../training_data/oversample_numpy_array.npy'
Oversample_Numpy_Array_File = '../training_data/oversample_numpy_bool_array.npy'
Oversample_Numpy_Array_Files = '../training_data/oversample_numpy_array_'
Statistics_Path = '../training_data/training_data_statistic'
Competition_Data = '../training_data/competition_onehot_nparray.npy'

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

''' assume one example like this
installed app onehot encoding [0, 0, 0, 0, 1]
installed apps onehot + onehot [0, 0, 0, 0, 1] + [0, 0, 0, 1, 0] = [0, 0, 0, 1, 1] dim = apps count
province onehot encoding [0, 0, 0, 1, 0, 0, ...., 0] dim = 26
computer brand onehot encoding [1, 0, 0, 0, 0] dim = brands count
x = [0, 0, 0, 1, 1, 0, 0, 0, 0, ..., 0, 1, 1, 0, 0, 0, 0]
     ^..apps.....^  ^......province.....^  ^....brand..^
y = 0, 1 or 1, 0
'''

def leaky_relu(x, alpha):
  return tf.maximum(alpha*x, x)

def load_statistics_file(file):
  dict = {}
  for key, val in csv.reader(open(file), encoding='utf-8'):
    dict[key] = val
  return dict

class Config(object):
  def __init__(self):
    self.batch_size = 256
    self.apps_dim = 100
    self.province_dim = 34
    self.computer_brand_dim = 20
    self.input_dim = self.computer_brand_dim + self.province_dim + self.apps_dim
    self.hidden_size_1 = 256
    self.hidden_size_2 = 128
    self.hidden_size_3 = 32
    self.output_class = 2
    self.max_epochs = 100
    self.early_stopping = 5
    self.dropout = 1.0
    self.lr = 0.01
    self.l2 = 0.0

    if Global_Debug == False:
      statistic_dict = load_statistics_file(Statistics_Path)
      self.computer_brand_dim = int(statistic_dict['brands count'])
      self.province_dim = int(statistic_dict['province count'])
      self.apps_dim = int(statistic_dict['applist max number'])
      # add applist unknow tag
      self.input_dim = self.apps_dim + self.province_dim + self.computer_brand_dim + 1

def generate_debug_data():
  # for debug purpose
  debug_sample = 4096
  debug_config = Config()
  # install apps onehot + onthot + ... + onehot
  # concat [sample, appsdim / 2 = random 0 1], [sample, appsdim / 2 = 0] = [sample, appsdim]
  dummy_data_input_apps_pos = np.concatenate(
    (np.random.randint(0, 2, size=(debug_sample, debug_config.apps_dim / 2)),
      np.zeros((debug_sample, debug_config.apps_dim / 2), dtype=np.int)), axis=1)
  print 'dummy_data_input_apps_pos.shape',dummy_data_input_apps_pos.shape
  # province onehot encoding, [debug_sample, province_dim]
  dummy_data_input_province_pos = np.zeros(
    (debug_sample, debug_config.province_dim), dtype=np.int)
  dummy_data_input_province_pos[
      np.arange(debug_sample), np.random.randint(0, debug_config.province_dim, debug_sample)] = 1
  print 'dummy_data_input_province_pos.shape',dummy_data_input_province_pos.shape
  # computer brand onehot encoding, [debug_sample, computer_brand_dim]
  dummy_data_input_computer_brand_pos = np.zeros(
    (debug_sample, debug_config.computer_brand_dim), dtype=np.int)
  dummy_data_input_computer_brand_pos[
      np.arange(debug_sample), np.random.randint(0, debug_config.computer_brand_dim, debug_sample)] = 1
  print 'dummy_data_input_computer_brand_pos.shape',dummy_data_input_computer_brand_pos.shape
  # concat [sample, appsdim / 2 = 0], [sample, appsdim / 2 = random 0 1] = [sample, appsdim]
  dummy_data_input_apps_neg = np.concatenate(
    (np.zeros((debug_sample, debug_config.apps_dim / 2), dtype=np.int),
      np.random.randint(0, 2, size=(debug_sample, debug_config.apps_dim / 2))), axis=1)
  # not care province and brand
  dummy_data_input_province_neg = dummy_data_input_province_pos
  dummy_data_input_computer_brand_neg = dummy_data_input_computer_brand_pos

  dummy_data_input_pos = np.concatenate(
    (dummy_data_input_apps_pos, dummy_data_input_province_pos, dummy_data_input_computer_brand_pos), axis=1)
  dummy_data_input_neg = np.concatenate(
    (dummy_data_input_apps_neg, dummy_data_input_province_neg, dummy_data_input_computer_brand_neg), axis=1)
  dummy_data_input = np.concatenate((dummy_data_input_pos, dummy_data_input_neg), axis=0)
  dummy_data_pos_label = np.concatenate((np.ones((debug_sample, 1)), np.zeros((debug_sample, 1))), axis=1)
  dummy_data_neg_label = np.concatenate((np.zeros((debug_sample, 1)), np.ones((debug_sample, 1))), axis=1)
  dummy_data_label = np.concatenate((dummy_data_pos_label, dummy_data_neg_label), axis=0)
  print 'one pos sample', dummy_data_input_pos[0]
  print 'one neg sample', dummy_data_input_neg[0]
  print 'one pos sample label', dummy_data_label[0]
  return dummy_data_input, dummy_data_label

def mapreduce_load_sample_nparray(files, i):
  print files + str(i) + '.npy'
  return np.load(files + str(i) + '.npy')

class NN_Baseline_Model():
  def load_data(self, debug=False):
    # for debug purpose
    sample_array_file = Undersample_Numpy_Array_File
    sample_array_files = Undersample_Numpy_Array_Files
    file_count = 10
    if Undersample_Array == False:
      sample_array_file = Oversample_Numpy_Array_File
      sample_array_files = Oversample_Numpy_Array_Files
      file_count = 15
    if debug:
      self.train_data = {}
      self.valid_data = {}
      self.test_data = {}
      dummy_data_input, dummy_data_label = generate_debug_data()
      self.train_data['input'] = dummy_data_input
      self.train_data['label'] = dummy_data_label
      self.valid_data['input'] = dummy_data_input
      self.valid_data['label'] = dummy_data_label
      self.test_data['input'] = dummy_data_input
      self.test_data['label'] = dummy_data_label
    else:
      if os.path.isfile(sample_array_file):
        print 'load npy file ', sample_array_file
        data = np.load(sample_array_file)
      else:
        print 'load npy files'
        #with ProcessPoolExecutor(5) as executor:
        map_ret = map(functools.partial(mapreduce_load_sample_nparray, sample_array_files), range(file_count))
        print 'concat '
        data = np.concatenate(tuple(array for array in map_ret), axis=0)
        print 'shuffle'
        np.random.shuffle(data)
        print 'save to single npy'
        np.save(sample_array_file, data)
      data_len = data.shape[0]
      print 'data_len', data_len
      train_data, valid_data, test_data = np.split(data, [int(data_len * 0.7), int(data_len * 0.85)], axis=0)
      print 'train_data len', train_data.shape[0]
      print 'valid_data len', valid_data.shape[0]
      print 'test_data len', test_data.shape[0]
      self.train_data = {}
      self.valid_data = {}
      self.test_data = {}
      self.train_data['label'], self.train_data['input'] = np.split(train_data, [2], axis=1)
      assert(self.train_data['label'].shape[1] == 2)
      assert(self.train_data['input'].shape[1] == self.config.input_dim)
      self.valid_data['label'], self.valid_data['input'] = np.split(valid_data, [2], axis=1)
      self.test_data['label'], self.test_data['input'] = np.split(test_data, [2], axis=1)
    #print self.train_data['input']
    #print self.train_data['label']
    
  def load_competition_data(self):
    self.competition_data = np.load(Competition_Data)

  def setup_placeholders(self):
    self.input_placeholder = tf.placeholder(
      tf.float32, shape=(None, self.config.input_dim), name='NN_Baseline_Input')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='NN_Baseline_Dropout')
    self.label_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.output_class), name='NN_Baseline_Label')
    self.phase_placeholder = tf.placeholder(tf.bool, name='phase')

  def build_loss_op(self, logits, labels):
    #print 'logits.shape', logits.get_shape()
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #print 'reg_losses.shape', reg_losses.get_shape()
    #print logits.shape
    #print labels.shape
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=1.0, name='NN_Baseline_CE_Per_Example')
    #print 'loss.shape', loss.get_shape()
    loss = tf.reduce_sum(loss, name='NN_Baseline_CE')
    #print 'reduce sum loss.shape', loss.get_shape()
    total_loss = reg_losses + loss
    #print 'total_loss.shape', total_loss.get_shape()
    #raw_input()
    #train_predict = tf.argmax(tf.exp(logits), 1)
    train_predict = tf.nn.softmax(logits)
    #print train_predict.shape
    return total_loss, train_predict

  def build_prediction(self, logits):
    #print logits.shape
    ret = tf.nn.softmax(logits)
    #print ret.shape
    return ret

  def build_training_op(self, loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
    return train_op

  def build_model(self, inputs, dropout, phase):

    with tf.variable_scope('NN_Baseline') as scope:
      h1 = tf.get_variable(
        'Hidden_Layer_1',
        [self.config.input_dim, self.config.hidden_size_1],
        tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(self.config.l2)
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
        regularizer=tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      b2 = tf.get_variable(
        'Bias_2',
        [self.config.hidden_size_2],
        tf.float32,
        tf.constant_initializer(0.0)
      )

      h3 = tf.get_variable(
        'Hidden_Layer_3',
        [self.config.hidden_size_2, self.config.hidden_size_3],
        tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      b3 = tf.get_variable(
        'Bias_3',
        [self.config.hidden_size_3],
        tf.float32,
        tf.constant_initializer(0.0)
      )

      output_weights = tf.get_variable(
        'Output_Layer',
        [self.config.hidden_size_3, self.config.output_class],
        tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(self.config.l2)
      )

      output_bias = tf.get_variable(
        'Bias_output',
        [self.config.output_class],
        tf.float32,
        tf.constant_initializer(0.0)
      )

    local1 = tf.matmul(tf.nn.dropout(inputs, dropout), h1) + b1
    # do not apply activation function in bn, use leakyrelu
    #bn1 = tf.contrib.layers.batch_norm(local1, center=True, scale=True, 
    #                                   is_training=phase, activation_fn=None,
    #                                   zero_debias_moving_mean=True,
    #                                   scope='bn1')
    activate1 = leaky_relu(local1, 0.1)
    
    local2 = tf.matmul(tf.nn.dropout(activate1, dropout), h2) + b2
    # do not apply activation function in bn, use leakyrelu
    #bn2 = tf.contrib.layers.batch_norm(local2, center=True, scale=True, 
    #                                   is_training=phase, activation_fn=None,
    #                                   zero_debias_moving_mean=True,
    #                                   scope='bn2')
    activate2 = leaky_relu(local2, 0.1)
    
    local3 = tf.matmul(tf.nn.dropout(activate2, dropout), h3) + b3
    # do not apply activation function in bn, use leakyrelu
    #bn3 = tf.contrib.layers.batch_norm(local3, center=True, scale=True, 
    #                                   is_training=phase, activation_fn=None,
    #                                   zero_debias_moving_mean=True,
    #                                   scope='bn3')
    activate3 = leaky_relu(local3, 0.1)
    
    linear_output = tf.matmul(tf.nn.dropout(activate3, dropout), output_weights) + output_bias    
    return linear_output

  def __init__(self, config):
    self.config = config
    if args.train or args.test or args.valid:
      self.load_data(Global_Debug)
    else:
      self.load_competition_data()
    self.setup_placeholders()
    self.linear_output = self.build_model(self.input_placeholder, self.dropout_placeholder, self.phase_placeholder)
    self.calculate_loss, self.train_predict = self.build_loss_op(self.linear_output, self.label_placeholder)
    self.predict_op = self.build_prediction(self.linear_output)
    self.train_op = self.build_training_op(self.calculate_loss)
    
  def infer(self, session, data):
    print 'phase infer ...'
    dp = 1.0
    train_phase = False
    predict_op = self.predict_op
    predict_result = None
    
    for step in xrange(0, data.shape[0], self.config.batch_size):
      x = data[step : step + self.config.batch_size]
      feed = {self.input_placeholder: x,
              self.dropout_placeholder: dp,
              self.phase_placeholder: train_phase}
      predict = session.run(predict_op, feed_dict=feed)
      if predict is not None:
        if predict_result is None:
          predict_result = predict
        else:
          predict_result = np.append(predict_result, predict, axis=0)
    #print 'predict_result.shape', predict_result.shape
    return predict_result

  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    total_loss = []
    predict_result = None
    predict_op = self.train_predict
    train_phase = True
    if train_op is None:
      print 'phase prediction ...'
      train_phase = False
      train_op = tf.no_op()
      predict_op = self.predict_op
      dp = 1.0
    else:
      print 'phase training ...'
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
              self.dropout_placeholder: dp,
              self.phase_placeholder: train_phase}
      loss, predict, _ = session.run(
        [self.calculate_loss, predict_op, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
          step, data['input'].shape[0], np.mean(total_loss)))
        sys.stdout.flush()
      if predict is not None:
        if predict_result is None:
          predict_result = predict
          #print 'predict.shape', predict.shape
          #raw_input()
        else:
          #print 'predict_result.shape', predict_result.shape
          #print 'predict.shape', predict.shape
          predict_result = np.append(predict_result, predict, axis=0)
          #print 'predict_result.shape', predict_result.shape
          #raw_input()
    if verbose:
      sys.stdout.write('\r\n')
    #print 'predict_result.shape', predict_result.shape
    return np.mean(total_loss), predict_result

def make_conf(labels, predictions):
  #print labels.shape
  #print predictions.shape
  labels = np.argmax(labels, axis=1)
  # if probability of male > threshold, like [[0.6, 0.4], [0.4, 0.6]] > 0.5 = [True, False]
  preds = predictions[:, 0] > args.threshold
  # to index (or argmax) [True, False] = [0, 1]
  preds = np.logical_not(preds).astype(np.int)
  #preds = np.argmax(predictions, axis=1)
  confmat = np.zeros([2, 2])
  for l,p in itertools.izip(labels, preds):
    confmat[l, p] += 1
  print confmat
  print 'tpr', confmat[0, 0] * 1.0 / (confmat[0, 0] + confmat[0, 1])
  print 'tnr', confmat[1, 1] * 1.0 / (confmat[1, 0] + confmat[1, 1])
  tp = confmat[0, 0]
  fp = confmat[1, 0]
  fn = confmat[0, 1]
  print 'f1=2tp/(2tp+fp+fn):', 2.0*tp/(2.0*tp + fp + fn)
  
def training():
  config = Config()
  model = NN_Baseline_Model(config)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  valid_save = tf.train.Saver()
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config) as session:
    best_val = float('inf')
    best_val_epoch = 0
    save_path = None
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print '** Epoch {}'.format(epoch)
      start = time.time()

      train_loss, prediction = model.run_epoch(
          session, model.train_data, train_op=model.train_op)
      print 'Training loss: {}'.format(train_loss)
      make_conf(model.train_data['label'], prediction)

      with tf.Session(config=session_config) as valid_session:
        valid_model_path = valid_save.save(session, './nn_baseline.valid_weights')
        valid_save.restore(valid_session, valid_model_path)
        valid_loss, prediction = model.run_epoch(valid_session, model.valid_data)
        print 'Validation loss: {}'.format(valid_loss)
        make_conf(model.valid_data['label'], prediction)

      if valid_loss < best_val:
        best_val = valid_loss
        best_val_epoch = epoch
        save_path = saver.save(session, './nn_baseline.best_weights')
        print('Model saved in file: %s' % save_path)
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)

    saver.restore(session, save_path)
    test_loss, test_predict = model.run_epoch(session, model.test_data)
    print '** Test loss: {}'.format(test_loss)
    make_conf(model.test_data['label'], test_predict)
    
def test():
  config = Config()
  model = NN_Baseline_Model(config)

  saver = tf.train.Saver()
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config) as session:
    saver.restore(session, './nn_baseline.best_weights')
    test_loss, test_predict = model.run_epoch(session, model.test_data)
    print '** Test loss: {}'.format(test_loss)
    make_conf(model.test_data['label'], test_predict)
    
def to_result_file(predict):
  j = 0
  result = []
  preds = predict[:, 0] > args.threshold
  preds = np.logical_not(preds).astype(np.int)
  for i in xrange(1200001, 1577085):
    if preds[j] == 0:
      s = str(i) + '\t' + 'male\n'
    else:
      s = str(i) + '\t' + 'female\n'
    result.append(s)
    j += 1
  with codecs.open('competition_result', 'w', 'utf-8') as f:
    f.writelines(result)
    
def infer():
  config = Config()
  model = NN_Baseline_Model(config)

  saver = tf.train.Saver()
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config) as session:
    saver.restore(session, './nn_baseline.best_weights')
    predict = model.infer(session, model.competition_data)
    print 'save result ...'
    to_result_file(predict)
    
def valid():
  config = Config()
  model = NN_Baseline_Model(config)

  saver = tf.train.Saver()
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config) as session:
    saver.restore(session, './nn_baseline.best_weights')
    test_loss, test_predict = model.run_epoch(session, model.valid_data)
    print '** Valid loss: {}'.format(test_loss)
    make_conf(model.valid_data['label'], test_predict)
if __name__ == "__main__":
  if args.train:
    training()
  elif args.test:
    test()
  elif args.valid:
    valid()
  elif args.infer:
    infer()
  else:
    print 'train or test or infere ?'