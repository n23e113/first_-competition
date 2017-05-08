﻿import sys
import pandas as pd
import codecs
import collections
import functools
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import unicodecsv as csv
import multiprocessing
import gc
import numpy as np

Global_Debug = False
Training_Data_File = 'aiwar_train_data'

if Global_Debug :
  Training_Data_File = '_data'
 
Tabbed_File = 'tabbed_training_data'
Undersample_Tabbed_File = 'undersample_tabbed_training_data'
Oversample_Tabbed_File = 'oversample_tabbed_training_data'
Undersample_Onehot_File = 'undersample_onehot_training_data'
Oversample_Onehot_File = 'oversample_onehot_training_data'
Undersample_Numpy_Array_File = 'undersample_numpy_array_'
Oversample_Numpy_Array_File = 'oversample_numpy_array_'
# 顺便统计一下训练数据的分类，记录到这个文件里面
Statistic_File = 'training_data_statistic'

# 统计少于100个的brand，都归到unknow类，数据太少，覆盖不到
Brand_Unknow_Threshold = 100
# 统计少于100个的app，归到unknow类
App_Unknow_Threshold = 100

def extract_features(line):
  features = line.split('\t')
  return features

# 把训练数据文件aiwar_train_data，格式化到文件 normalized_training_data
def distinct_features_from_org_training_data():
  brand_dict = {}
  apps_dict = {}
  province_dict = {}
  gender_dict = {}
  
  #lines = [line.rstrip('\r\n') for line in open(Training_Data_File)]
  print('read aiwar_train_data ...')
  sys.stdout.write('*')
  sys.stdout.flush()
  lines = [line.rstrip('\r\n') for line in codecs.open(Training_Data_File, 'r', 'utf-8')]
  count = 0
  sys.stdout.write('*')
  sys.stdout.flush()
  for line in lines:
    features = extract_features(line)
    if len(features) != 5 :
      print count
      print features
      assert(len(features) == 5)
    
    if brand_dict.get(features[2]) == None:
      brand_dict[features[2]] = 1
    else:
      brand_dict[features[2]] += 1
    
    if gender_dict.get(features[1]) == None:
      gender_dict[features[1]] = 1
    else:
      gender_dict[features[1]] += 1
    
    if province_dict.get(features[4]) == None:
      province_dict[features[4]] = 1
    else:
      province_dict[features[4]] += 1
      
    apps_list = features[3].split(',')
    for app in apps_list:
      if apps_dict.get(int(app)) == None:
        apps_dict[int(app)] = 1
      else:
        apps_dict[int(app)] = apps_dict[int(app)] + 1
        
    count += 1
    if count % 100000 == 0:
        sys.stdout.write('*')
        sys.stdout.flush()
  sys.stdout.write('\n')
  return gender_dict, brand_dict, collections.OrderedDict(sorted(apps_dict.items())), province_dict, lines
  
def from_normalized_file_to_original_data(file, gender_dict, brand_dict, apps_dict, province_dict):
  lines = [line.rstrip('\r\n') for line in codecs.open(file, 'r', 'utf-8')]
  max_key = apps_dict.keys()[len(apps_dict) - 1]
  print 'max_key', max_key
  ret = []
  for line in lines:
    features = extract_features(line)
    s = ''
    for i in xrange(max_key):
      if int(features[i + 4]) == 1:
        s += str(i + 1) + ','
      if i == max_key:
        s = s[:-1] + '\t'
    ret.append(features[0] + '\t' + features[1] + '\t' + features[2] + '\t' + s + features[3])
  for line in ret:
    print line
    
def save_statistics_file(file, dict):
  w = csv.writer(codecs.open(file, 'w'), encoding='utf-8')
  tolist = sorted(dict.items(), key=lambda x: x[1], reverse=True)
  for p in tolist:
    w.writerow([p[0], p[1]])
    
def load_statistics_file(file):
  dict = {}
  for key, val in csv.reader(open(file), encoding='utf-8'):
    dict[key] = val
  return dict
    
def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in xrange(0, len(l), n):
    yield l[i:i + n]
    
# python file.writelines空间效率好像有问题，分段些
def string_list_to_file(file, string_list):
  with codecs.open(file, 'w', 'utf-8') as f:
    chunk = chunks(string_list, 10000)
    for c in chunk:
     f.writelines(c)
     gc.collect()
 
def condiction_on_gender(gender_dict, brand_dict, apps_dict, province_dict, lines):
  male_brand_dict = dict.fromkeys(brand_dict, 0)
  male_apps_dict = dict.fromkeys(apps_dict, 0)
  male_province_dict = dict.fromkeys(province_dict, 0)
  
  female_brand_dict = dict.fromkeys(brand_dict, 0)
  female_apps_dict = dict.fromkeys(apps_dict, 0)
  female_province_dict = dict.fromkeys(province_dict, 0)
  
  count = 0
  for line in lines:
    features = extract_features(line)
    if features[1] == u'male':
      male_brand_dict[features[2]] += 1
      male_province_dict[features[4]] += 1
      apps_list = features[3].split(',')
      for app in apps_list:
        male_apps_dict[int(app)] += 1
    else:
      female_brand_dict[features[2]] += 1
      female_province_dict[features[4]] += 1
      apps_list = features[3].split(',')
      for app in apps_list:
        female_apps_dict[int(app)] += 1
    count += 1
    if count % 100000 == 0:
        sys.stdout.write('+')
        sys.stdout.flush()
  sys.stdout.write('\n')
  return male_brand_dict, male_apps_dict, male_province_dict, female_brand_dict, female_apps_dict, female_province_dict
    
def to_tabbed_file():
  gender_dict, brand_dict, apps_dict, province_dict, lines = distinct_features_from_org_training_data()
  
  write_to_file(True, gender_dict, brand_dict, apps_dict, province_dict, lines)
  gc.collect()
  write_to_file(False, gender_dict, brand_dict, apps_dict, province_dict, lines)
  gc.collect()
  
  print '^'
  
  if Global_Debug:
    from_normalized_file_to_original_data(Tabbed_File, gender_dict, brand_dict, apps_dict, province_dict)
    
  save_statistics_file(Statistic_File + '_brands', brand_dict)
  save_statistics_file(Statistic_File + '_gender', gender_dict)
  save_statistics_file(Statistic_File + '_province', province_dict)
  save_statistics_file(Statistic_File + '_applist', apps_dict)
  
  statistic_dict = {}
  statistic_dict['brands count'] = len(brand_dict)
  statistic_dict['gender count'] = len(gender_dict)
  statistic_dict['province count'] = len(province_dict)
  statistic_dict['applist count'] = len(apps_dict)
  statistic_dict['applist max number'] = apps_dict.keys()[len(apps_dict) - 1]
  save_statistics_file(Statistic_File, statistic_dict)
  print '^'
  
  # 看看以性别为前提的分布
  male_brand_dict, male_apps_dict, male_province_dict, female_brand_dict, female_apps_dict, female_province_dict = condiction_on_gender(
    gender_dict, brand_dict, apps_dict, province_dict, lines)
  save_statistics_file(Statistic_File + '_brands_male', male_brand_dict)
  save_statistics_file(Statistic_File + '_province_male', male_province_dict)
  save_statistics_file(Statistic_File + '_applist_male', male_apps_dict)
  save_statistics_file(Statistic_File + '_brands_female', female_brand_dict)
  save_statistics_file(Statistic_File + '_province_female', female_province_dict)
  save_statistics_file(Statistic_File + '_applist_female', female_apps_dict)
  print '^'
  
def write_to_file(onehot, gender_dict, brand_dict, apps_dict, province_dict, lines):
  chunk_list = chunks(lines, 50000)
  with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
    map_ret = executor.map(
      functools.partial(mapreduce_to_tabbed_file, onehot, gender_dict, brand_dict, apps_dict, province_dict), chunk_list)
  sys.stdout.write('\n')
  sys.stdout.flush()
  
  if onehot == False:
    male_data = []
    female_data = []
    ret_list = []
    for l in map_ret:
      ret_list += l
    sys.stdout.write('+')
    sys.stdout.flush()
    if Global_Debug:
      print ret_list[0]
    string_list_to_file(Tabbed_File, ret_list)
    sys.stdout.write('-')
    sys.stdout.flush()
    #with codecs.open(Tabbed_File, 'w', 'utf-8') as f:
    #  if Global_Debug:
    #    print ret_list[0]
    #  f.writelines(ret_list)
      
    for l in ret_list:
      features = extract_features(l)
      if features[1] == u'male':
        male_data.append(l)
      else:
        female_data.append(l)
    ret_list = []
    gc.collect()
    undersample_data = []
    sys.stdout.write('+')
    sys.stdout.flush()
    for i in xrange(min(len(male_data), len(female_data))):
      undersample_data.append(male_data[i])
      undersample_data.append(female_data[i])
    sys.stdout.write('+')
    sys.stdout.flush()  
    string_list_to_file(Undersample_Tabbed_File, undersample_data)
    sys.stdout.write('-')
    sys.stdout.flush()
    #with codecs.open(Undersample_Tabbed_File, 'w', 'utf-8') as f:
    #  f.writelines(undersample_data)
    undersample_data = []
    gc.collect()
    oversample_data = []
    for i in xrange(max(len(male_data), len(female_data))):
      oversample_data.append(male_data[i % len(male_data)])
      oversample_data.append(female_data[i % len(female_data)])
    sys.stdout.write('+')
    sys.stdout.flush()
    string_list_to_file(Oversample_Tabbed_File, oversample_data)
    sys.stdout.write('-')
    sys.stdout.flush()
    #with codecs.open(Oversample_Tabbed_File, 'w', 'utf-8') as f:
    #  f.writelines(oversample_data)
    oversample_data = []
    gc.collect()

  else:
    male_onehot = []
    female_onehot = []
    onehot_list = []
    for l in map_ret:
      onehot_list += l
    for l in onehot_list:
      features = extract_features(l)
      if features[1] == '1':
        male_onehot.append(l)
      else:
        female_onehot.append(l)
    onehot_list = []
    gc.collect()
    undersample_onehot = []
    sys.stdout.write('+')
    sys.stdout.flush()
    for i in xrange(min(len(male_onehot), len(female_onehot))):
      undersample_onehot.append(male_onehot[i])
      undersample_onehot.append(female_onehot[i])
    sys.stdout.write('+')
    sys.stdout.flush()
    string_list_to_file(Undersample_Onehot_File, undersample_onehot)
    sys.stdout.write('-')
    sys.stdout.flush()
    #with codecs.open(Undersample_Onehot_File, 'w', 'utf-8') as f:
    #  f.writelines(undersample_onehot)
    oversample_onehot = []
    for i in xrange(max(len(male_onehot), len(female_onehot))):
      oversample_onehot.append(male_onehot[i % len(male_onehot)])
      oversample_onehot.append(female_onehot[i % len(female_onehot)])
    sys.stdout.write('+')
    sys.stdout.flush()
    string_list_to_file(Oversample_Onehot_File, oversample_onehot)
    sys.stdout.write('-')
    sys.stdout.flush()
    #with codecs.open(Oversample_Onehot_File, 'w', 'utf-8') as f:
    #  f.writelines(oversample_onehot)  
  
def mapreduce_to_tabbed_file(onehot, gender_dict, brand_dict, apps_dict, province_dict, lines):
  #print 'gender', len(gender_dict), gender_dict
  #print 'brand', len(brand_dict)
  #print 'app list', len(apps_dict), apps_dict.keys()[len(apps_dict) - 1]
  #print sorted(apps_dict.items(), key=lambda x: x[1])
  #print 'province', len(province_dict)
  #for province, _ in province_dict.iteritems():
  #  print province
  #print('generate normalized data ...')
  brand_to_index = {}
  index = 0
  for k, _ in brand_dict.iteritems():
    brand_to_index[k] = index
    index += 1
  
  index = 0
  province_to_index = {}
  for k, _ in province_dict.iteritems():
    province_to_index[k] = index
    index += 1
  
  max_key = apps_dict.keys()[len(apps_dict) - 1]
  count = 0
  feature_out = []
  # 下面这个是针对nparray文件用的
  onehot_out = []
  for line in lines:
    features = extract_features(line)
    # applist onehot
    apps_list = features[3].split(',')
    s = '0\t' * (max_key)
    s = list(s)
    for app in apps_list:
      # app 编号从1开始
      s[(int(app) - 1) * 2] = '1'
    s = ''.join(s)
    if onehot == False:
      feature_out.append(''.join([features[0], '\t', features[1], '\t', features[2], '\t', features[4], '\t', s[:-1], '\n']))
    else:
      # gender onehot
      gender_onehot = '1\t0\t'
      if features[1] == u'female':
        gender_onehot = '0\t1\t'
        
      # brand onehot
      brand_dict_len = len(brand_dict)
      brand_onehot = '0\t' * brand_dict_len
      brand_onehot = list(brand_onehot)
      brand_onehot[brand_to_index[features[2]] * 2] = '1'
      brand_onehot = ''.join(brand_onehot)
      
      # province onehot
      province_dict_len = len(province_dict)
      province_onehot = '0\t' * province_dict_len
      province_onehot = list(province_onehot)
      province_onehot[province_to_index[features[4]] * 2] = '1'
      province_onehot = ''.join(province_onehot)
      
      onehot_out.append(''.join([features[0], '\t', gender_onehot, brand_onehot, province_onehot, s[:-1], '\n']))
    
    count += 1
    if count % 100 == 0:
      gc.collect()
  sys.stdout.write('>')
  sys.stdout.flush()
  
  if onehot:
    return onehot_out
  else:
    return feature_out

# 从格式化好的文件，读取数据
# 注意
# 1.格式化文件内，性别，机器型号，省份是没有编码的，需要在这个函数内用onehot编码
# 2.tab文件内，第一列是编号，没有作用，需要去掉
def mapreduce_to_nparray_file(app_max, brands_count, gender_count, province_count, i):
  doc = codecs.open('split/undersample_onehot_split_0' + str(i),'r','utf-8')
  df = pd.read_csv(doc, sep='\t', header=None)
  array = df.as_matrix()
  if Global_Debug:
    print array
  # id 
  assert array.shape[1] == app_max + brands_count + province_count + gender_count + 1
  print Undersample_Numpy_Array_File + str(i)
  np.save(Undersample_Numpy_Array_File + str(i), np.split(array, [1], axis=1)[1])
  if Global_Debug:
    print np.load(Undersample_Numpy_Array_File)
  return i

def tabbed_file_to_nparray_file():
  statistics_dict = load_statistics_file(Statistic_File)
  app_max = int(statistics_dict['applist max number'])
  brands_count = int(statistics_dict['brands count'])
  gender_count = int(statistics_dict['gender count'])
  province_count = int(statistics_dict['province count'])
  
  with ProcessPoolExecutor(5) as executor:
    map_ret = executor.map(
      functools.partial(mapreduce_to_nparray_file, app_max, brands_count, gender_count, province_count), range(10))
  print map_ret

if __name__ == "__main__":
  #to_tabbed_file()
  tabbed_file_to_nparray_file()