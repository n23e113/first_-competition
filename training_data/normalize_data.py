import sys
import pandas as pd
import codecs
import collections
import functools
from concurrent.futures import ProcessPoolExecutor
import unicodecsv as csv
import multiprocessing

Global_Debug = False
Training_Data_File = 'aiwar_train_data'

if Global_Debug :
  Training_Data_File = '_data'
 
Normalized_File = 'normalized_training_data'
# 顺便统计一下训练数据的分类，记录到这个文件里面
Statistic_File = 'training_data_statistic'

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
    for i in xrange(max_key + 1):
      if int(features[i + 4]) == 1:
        s += str(i) + ','
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
    
def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in xrange(0, len(l), n):
    yield l[i:i + n]
    
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
    
def to_normalized_file():
  gender_dict, brand_dict, apps_dict, province_dict, lines = distinct_features_from_org_training_data()
  chunk_list = chunks(lines, 50000)
    
  with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
    ret_list = executor.map(
      functools.partial(map_to_normalized_file, gender_dict, brand_dict, apps_dict, province_dict), chunk_list)
  sys.stdout.write('\n')
  sys.stdout.flush()
  
  with codecs.open(Normalized_File, 'w', 'utf-8') as f:
    for l in ret_list:
      if Global_Debug:
        print l[0]
      f.writelines(l)
      
  if Global_Debug:
    from_normalized_file_to_original_data(Normalized_File, gender_dict, brand_dict, apps_dict, province_dict)
    
  save_statistics_file(Statistic_File + '_brands', brand_dict)
  save_statistics_file(Statistic_File + '_gender', gender_dict)
  save_statistics_file(Statistic_File + '_province', province_dict)
  save_statistics_file(Statistic_File + '_applist', apps_dict)
  with open(Statistic_File, 'w') as f:
    f.write('brands count ' + str(len(brand_dict)) + '\n')
    f.write('gender count ' + str(len(gender_dict)) + '\n')
    f.write('province count ' + str(len(province_dict)) + '\n')
    f.write('applist count ' + str(len(apps_dict)) + '\n')
    f.write('applist max number ' + str(apps_dict.keys()[len(apps_dict) - 1]) + '\n')
  
  # 看看以性别为前提的分布
  male_brand_dict, male_apps_dict, male_province_dict, female_brand_dict, female_apps_dict, female_province_dict = condiction_on_gender(
    gender_dict, brand_dict, apps_dict, province_dict, lines)
  save_statistics_file(Statistic_File + '_brands_male', male_brand_dict)
  save_statistics_file(Statistic_File + '_province_male', male_province_dict)
  save_statistics_file(Statistic_File + '_applist_male', male_apps_dict)
  save_statistics_file(Statistic_File + '_brands_female', female_brand_dict)
  save_statistics_file(Statistic_File + '_province_female', female_province_dict)
  save_statistics_file(Statistic_File + '_applist_female', female_apps_dict)
  
def map_to_normalized_file(gender_dict, brand_dict, apps_dict, province_dict, lines):
  #print 'gender', len(gender_dict), gender_dict
  #print 'brand', len(brand_dict)
  #print 'app list', len(apps_dict), apps_dict.keys()[len(apps_dict) - 1]
  #print sorted(apps_dict.items(), key=lambda x: x[1])
  #print 'province', len(province_dict)
  #for province, _ in province_dict.iteritems():
  #  print province
  #print('generate normalized data ...')
  max_key = apps_dict.keys()[len(apps_dict) - 1]
  count = 0
  feature_out = []
  for line in lines:
    features = extract_features(line)
    apps_list = features[3].split(',')
    s = '0\t' * (max_key + 1)
    s = list(s)
    #print len(s)
    for app in apps_list:
      #print int(app)
      s[int(app) * 2] = '1'
    '''
    for i in xrange(max_key + 1):
      has_app = False
      for app in apps_list:
        if int(app) == i:
          s += '1\t'
          has_app = True
          break
      if has_app == False:
        s += '0\t'
      else:
        has_app = False
    '''
    s = ''.join(s)
    feature_out.append(''.join([features[0], '\t', features[1], '\t', features[2], '\t', features[4], '\t', s[:-1], '\n']))
    count += 1
  '''
      if count % 1000 == 0:
        f.writelines(feature_out)
        feature_out[:] = []
        sys.stdout.write('>')
        sys.stdout.flush()
    f.writelines(feature_out)
  '''
  sys.stdout.write('>')
  sys.stdout.flush()
  return feature_out

# 从格式化好的文件，读取数据，返回nparray
def normalize_data_from_file():
  ret = {}
  
  pd.read
  
  return 

if __name__ == "__main__":
  to_normalized_file()