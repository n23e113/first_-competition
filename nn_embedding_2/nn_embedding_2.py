import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import os.path
import getpass
import sys
import time
import numpy as np
from copy import deepcopy
import itertools
import tensorflow as tf
import unicodecsv as csv
from concurrent.futures import ProcessPoolExecutor

sys.path.append("..")
from q2_initialization import xavier_weight_init

tf.logging.set_verbosity(tf.logging.ERROR)

Global_Debug = False
Undersample_Numpy_Array_File = '../training_data/undersample_numpy_array.npy'
Undersample_Numpy_Array_Files = '../training_data/undersample_numpy_array_'
Statistics_Path = '../training_data/training_data_statistic'

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

''' 
assume one example like this
installed app onehot encoding [0, 0, 0, 0, 1]
installed apps onehot + onehot [0, 0, 0, 0, 1] + [0, 0, 0, 1, 0] = [0, 0, 0, 1, 1] dim = apps count
province onehot encoding [0, 0, 0, 1, 0, 0, ...., 0] dim = 26
computer brand onehot encoding [1, 0, 0, 0, 0] dim = brands count
x = [0, 0, 0, 1, 1, 0, 0, 0, 0, ..., 0, 1, 1, 0, 0, 0, 0]
     ^..apps.....^  ^......province.....^  ^....brand..^
y = 0, 1 or 1, 0

installed apps onehots map to embedding example
assume:
apps count = 5
apps type = A, B, C, D, E
embedding dim = 3
onehot: [1, 0, 0, 1, 1], A, D, E installed
embedding initial:
 [[0.5, 0.4, 0.3],
  [0.6, 0.5, 0.4],
  [0.5, 0.5, 0.3],
  [0.5, 0.6, 0.7],
  [0.7, 0.4, 0.3]]
map:
 [[0.5, 0.4, 0.3],   [[0.5, 0.4, 0.3],                  [[1, 1, 1],
  [0.0, 0.0, 0.0],    [0.6, 0.5, 0.4],                   [0, 0, 0],
  [0.0, 0.0, 0.0], =  [0.5, 0.5, 0.3], hadamard product  [0, 0, 0],
  [0.5, 0.6, 0.7],    [0.5, 0.6, 0.7],                   [1, 1, 1],
  [0.7, 0.4, 0.3]]    [0.7, 0.4, 0.3]]                   [1, 1, 1]]
  ^....nn input..^   ^embedding matrix^                 ^onehot expand^
                       tf.variable                       tf.placeholder
  
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
    self.province_onehot_dim  = 34
    self.computer_brand_onehot_dim = 20
    self.input_dim = self.apps_dim + self.province_onehot_dim  + self.computer_brand_onehot_dim
    self.province_embedding_dim = 8
    self.computer_brand_embedding_dim = 16
    self.apps_nn_output_dim = 16
    self.hidden_input_dim = self.computer_brand_embedding_dim + self.province_embedding_dim + self.apps_nn_output_dim
    self.hidden_size_1 = 16
    self.hidden_size_2 = 8
    self.output_class = 2
    self.max_epochs = 10
    self.early_stopping = 2
    self.dropout = 1.0
    self.lr = 0.01
    self.l2 = 0.0
    
    if Global_Debug == False:
      statistic_dict = load_statistics_file(Statistics_Path)
      self.computer_brand_onehot_dim = int(statistic_dict['brands count'])
      self.province_onehot_dim  = int(statistic_dict['province count'])
      self.apps_dim = int(statistic_dict['applist max number'])
      self.input_dim = self.computer_brand_onehot_dim + self.province_onehot_dim + self.apps_dim
      self.hidden_input_dim = self.computer_brand_embedding_dim + self.province_embedding_dim + self.apps_nn_output_dim
      print self.input_dim, self.hidden_input_dim

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

def mapreduce_load_nparray(i):
  print i
  return np.load(Undersample_Numpy_Array_Files + str(i) + '.npy')
  
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
      if os.path.isfile(Undersample_Numpy_Array_File):
        print 'load npy file ...'
        data = np.load(Undersample_Numpy_Array_File)
      else:
        print 'load npy files ...'
        #with ProcessPoolExecutor(5) as executor:
        map_ret = map(mapreduce_load_nparray, range(10))
        data = np.concatenate(tuple(array for array in map_ret), axis=0)
        np.random.shuffle(data)
        np.save(Undersample_Numpy_Array_File, data)
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
      assert(self.train_data['input'].shape[1] == config.input_dim)
      self.valid_data['label'], self.valid_data['input'] = np.split(valid_data, [2], axis=1)
      self.test_data['label'], self.test_data['input'] = np.split(test_data, [2], axis=1)
    #print self.train_data['input']
    #print self.train_data['label']

  def setup_placeholders(self):
    self.input_placeholder = tf.placeholder(
      tf.float32, shape=(None, self.config.input_dim), name='NN_Baseline_Input')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='NN_Baseline_Dropout')
    self.label_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.output_class), name='NN_Baseline_Label')

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
    train_predict = tf.argmax(tf.exp(logits), 1)
    return total_loss, train_predict

  def build_prediction(self, logits):
    return tf.argmax(tf.exp(logits), 1)

  def build_training_op(self, loss):
    train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
    return train_op

  def build_model(self, inputs, dropout):

    computer_brand_onehot, province_onehot, apps_bitmap = tf.split(
      inputs, [config.computer_brand_onehot_dim, config.province_onehot_dim, config.apps_dim], 1)
    print apps_bitmap.shape, province_onehot.shape, computer_brand_onehot.shape
      
    with tf.variable_scope('NN_Embedding') as scope:
      with tf.device('/cpu:0'):
        province_embedding = tf.get_variable(
          "Province_Embedding",
          [self.config.province_onehot_dim, self.config.province_embedding_dim],
          tf.float32,
          xavier_weight_init())
        province = tf.nn.embedding_lookup(
          province_embedding, tf.argmax(province_onehot, axis=1))
        print province.shape
          
      with tf.device('/cpu:0'):
        computer_brand_embedding = tf.get_variable(
          "Computer_Brand_Embedding",
          [self.config.computer_brand_onehot_dim, self.config.computer_brand_embedding_dim],
          tf.float32,
          xavier_weight_init())
        computer_brand = tf.nn.embedding_lookup(
          computer_brand_embedding, tf.argmax(computer_brand_onehot, axis=1))
        print computer_brand.shape
  
    with tf.variable_scope('NN_Baseline') as scope:
      h1 = tf.get_variable(
        'Hidden_Layer_1',
        [self.config.hidden_input_dim, self.config.hidden_size_1],
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
      
      apps_nn_weights = tf.get_variable(
        'Apps_nn_weights',
        [self.config.apps_dim, self.config.apps_nn_output_dim],
        tf.float32,
        xavier_weight_init(),
        tf.contrib.layers.l2_regularizer(self.config.l2)
      )
      
      apps_nn_bias = tf.get_variable(
        'Apps_nn_bias',
        [self.config.apps_nn_output_dim],
        tf.float32,
        tf.constant_initializer(0.0)
      )
      
    apps_nn_output = leaky_relu(tf.matmul(tf.nn.dropout(apps_bitmap, dropout), apps_nn_weights) + apps_nn_bias, 0.1)
    print tf.concat([computer_brand, province, apps_nn_output], 1).shape
    local1 = leaky_relu(tf.matmul(tf.nn.dropout(
      tf.concat([computer_brand, province, apps_nn_output], 1), dropout), h1) + b1, 0.1)
    local2 = leaky_relu(tf.matmul(tf.nn.dropout(local1, dropout), h2) + b2, 0.1)
    linear_output = tf.matmul(tf.nn.dropout(local2, dropout), output_weights) + output_bias

    return linear_output

  def __init__(self, config):
    self.config = config
    self.load_data(Global_Debug)
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
    if train_op is None:
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
      if predict is not None:
        if predict_result is None:
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
  labels = np.argmax(labels, axis=1)
  #print labels.shape
  #print predictions.shape
  confmat = np.zeros([2, 2])
  for l,p in itertools.izip(labels, predictions):
    confmat[l, p] += 1
  print confmat
  print 'tpr:', confmat[0, 0] * 1.0 / (confmat[0, 0] + confmat[0, 1])
  print 'tnr:', confmat[1, 1] * 1.0 / (confmat[1, 0] + confmat[1, 1])
  tp = confmat[0, 0]
  fp = confmat[1, 0]
  fn = confmat[0, 1]
  print 'f1=2tp/(2tp+fp+fn):', 2.0*tp/(2.0*tp + fp + fn)

if __name__ == "__main__":
  config = Config()
  model = NN_Baseline_Model(config)

  init = tf.initialize_all_variables()
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
        valid_loss, prediction = model.run_epoch(session, model.valid_data)
        print 'Validation loss: {}'.format(valid_loss)
        make_conf(model.valid_data['label'], prediction)
        
      if valid_loss < best_val:
        print '** {} < {}'.format(valid_loss, best_val)
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
