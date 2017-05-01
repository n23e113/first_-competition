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

''' assume one example like this
installed app onehot encoding [0, 0, 0, 0, 1]
installed apps onehot + onehot [0, 0, 0, 0, 1] + [0, 0, 0, 1, 0] = [0, 0, 0, 1, 1] dim = apps count
province onehot encoding [0, 0, 0, 1, 0, 0, ...., 0] dim = 26
computer brand onehot encoding [1, 0, 0, 0, 0] dim = brands count
x = [0, 0, 0, 1, 1, 0, 0, 0, 0, ..., 0, 1, 1, 0, 0, 0, 0]
     ^..apps.....^  ^......province.....^  ^....brand..^
y = 0 or 1

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

class Config(object):
  batch_size = 64
  apps_dim = 100
  province_onehot_dim = 34
  computer_brand_onehot_dim = 20
  input_dim = apps_dim + province_onehot_dim + computer_brand_onehot_dim
  province_embedding_dim = 4
  computer_brand_embedding_dim = 8
  hidden_input_dim = apps_dim + province_embedding_dim + computer_brand_embedding_dim
  hidden_size_1 = 100
  hidden_size_2 = 100
  output_class = 2
  max_epochs = 10
  early_stopping = 2
  dropout = 0.5
  lr = 0.001
  l2 = 0.0

# for debug purpose
debug_sample = 4096
debug_config = Config()

def generate_pos_data():
  # install apps onehot + onthot + ... + onehot
  # concat [sample, appsdim / 2 = random 0 1], [sample, appsdim / 2 = 0] = [sample, appsdim]
  dumy_data_input_apps_pos = np.concatenate(
    (np.random.randint(0, 2, size=(debug_sample, debug_config.apps_dim / 2)),
      np.zeros((debug_sample, debug_config.apps_dim / 2), dtype=np.int)), axis=1)
  print 'dumy_data_input_apps_pos.shape',dumy_data_input_apps_pos.shape
  # province onehot encoding, [debug_sample, province_dim]
  dumy_data_input_province_pos = np.zeros(
    (debug_sample, debug_config.province_onehot_dim), dtype=np.int)
  dumy_data_input_province_pos[
      np.arange(debug_sample), np.random.randint(0, debug_config.province_onehot_dim, debug_sample)] = 1
  print 'dumy_data_input_province_pos.shape',dumy_data_input_province_pos.shape
  # computer brand onehot encoding, [debug_sample, computer_brand_dim]
  dumy_data_input_computer_brand_pos = np.zeros(
    (debug_sample, debug_config.computer_brand_onehot_dim), dtype=np.int)
  dumy_data_input_computer_brand_pos[
      np.arange(debug_sample), np.random.randint(0, debug_config.computer_brand_onehot_dim, debug_sample)] = 1
  print 'dumy_data_input_computer_brand_pos.shape',dumy_data_input_computer_brand_pos.shape
  
  return np.concatenate(
    (dumy_data_input_apps_pos, dumy_data_input_province_pos, dumy_data_input_computer_brand_pos), axis=1)
  
def generate_neg_data():
  # concat [sample, appsdim / 2 = 0], [sample, appsdim / 2 = random 0 1] = [sample, appsdim]
  dumy_data_input_apps_neg = np.concatenate(
    (np.zeros((debug_sample, debug_config.apps_dim / 2), dtype=np.int),
      np.random.randint(0, 2, size=(debug_sample, debug_config.apps_dim / 2))), axis=1)
  # not care province and brand
  dumy_data_input_province_neg = np.zeros(
    (debug_sample, debug_config.province_onehot_dim), dtype=np.int)
  dumy_data_input_province_neg[
      np.arange(debug_sample), np.random.randint(0, debug_config.province_onehot_dim, debug_sample)] = 1
      
  dumy_data_input_computer_brand_neg = np.zeros(
    (debug_sample, debug_config.computer_brand_onehot_dim), dtype=np.int)
  dumy_data_input_computer_brand_neg[
      np.arange(debug_sample), np.random.randint(0, debug_config.computer_brand_onehot_dim, debug_sample)] = 1
      
  return np.concatenate(
    (dumy_data_input_apps_neg, dumy_data_input_province_neg, dumy_data_input_computer_brand_neg), axis=1)

dumy_data_input_pos = generate_pos_data()
dumy_data_input_neg = generate_neg_data()
dumy_data_input = np.concatenate((dumy_data_input_pos, dumy_data_input_neg), axis=0)
dumy_data_label = np.concatenate((np.ones(debug_sample, dtype=np.int), np.zeros(debug_sample, dtype=np.int)), axis=0)

dumy_data_test_pos = generate_pos_data()
dumy_data_test_neg = generate_neg_data()
dumy_data_test_input = np.concatenate((dumy_data_test_pos, dumy_data_test_neg), axis=0)
dumy_data_test_label = np.concatenate((np.ones(debug_sample, dtype=np.int), np.zeros(debug_sample, dtype=np.int)), axis=0)

print 'one pos sample', dumy_data_input_pos[0]
print 'one neg sample', dumy_data_input_neg[0]

class NN_Embedding_Model():

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
    self.test_data['input'] = dumy_data_test_input
    self.test_data['label'] = dumy_data_test_label
    #print self.train_data['input']
    #print self.train_data['label']

  def setup_placeholders(self):
    self.input_placeholder = tf.placeholder(
      tf.float32, shape=(None, self.config.input_dim), name='NN_Embedding_Input')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='NN_Embedding_Dropout')
    self.label_placeholder = tf.placeholder(tf.int32, shape=(None), name='NN_Embedding_Label')

  def build_loss_op(self, logits, labels):
    #print 'logits.shape', logits.get_shape()
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #print 'reg_losses.shape', reg_losses.get_shape()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='NN_Embedding_CE_Per_Example')
    #print 'loss.shape', loss.get_shape()
    loss = tf.reduce_sum(loss, name='NN_Embedding_CE')
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

    apps_onehot, province_onehot, computer_brand_onehot = tf.split(
      inputs, [config.apps_dim, config.province_onehot_dim, config.computer_brand_onehot_dim], 1)
      
    with tf.variable_scope('NN_Embedding') as scope:
      with tf.device('/cpu:0'):
        province_embedding = tf.get_variable(
          "Province_Embedding",
          [self.config.province_onehot_dim, self.config.province_embedding_dim],
          tf.float32,
          xavier_weight_init())
        province = tf.nn.embedding_lookup(
          province_embedding, tf.argmax(province_onehot, axis=1))
          
      with tf.device('/cpu:0'):
        computer_brand_embedding = tf.get_variable(
          "Computer_Brand_Embedding",
          [self.config.computer_brand_onehot_dim, self.config.computer_brand_embedding_dim],
          tf.float32,
          xavier_weight_init())
        computer_brand = tf.nn.embedding_lookup(
          computer_brand_embedding, tf.argmax(computer_brand_onehot, axis=1))
    
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


    local1 = tf.nn.relu(tf.matmul(tf.nn.dropout(
      tf.concat([apps_onehot, province, computer_brand], 1), dropout), h1) + b1)
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
  model = NN_Embedding_Model(config)

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
        save_path = saver.save(session, './nn_embedding.weights')
        print('Model saved in file: %s' % save_path)
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)

    saver.restore(session, save_path)
    test_loss, test_predict = model.run_epoch(session, model.test_data)
    print 'Test loss: {}'.format(test_loss)
    print make_conf(model.test_data['label'], test_predict)
