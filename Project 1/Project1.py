import time
import sys
import csv
import numpy as np
# ignores warning for row with entropy for zero
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold

params = {0:'dia', 1:'eda', 2:'sys', 3:'res'}

# read in data. if the rows value is in index list (meaning we want it), append appropiate things
def load_data(index):
  ids = []
  data = []
  with open('Project1Data.csv', 'r') as file:
    reader = csv.reader(file)
    for i,row in enumerate(reader):
      if(i % 4) in index:          
        ids.append(row[0]),
        data.append(np.array(row[3::]).astype(np.float))
      #fomatting to know were not stalling
      if (i+1) % 20 == 0:
        print('READING COLUMNS',i-19,'-',i)
  print()
  return ids, data

def calculate_features(data):
  print("CRAFTING FEATURES...")
  print()
  #if we have all, then mod 4 to add, else mod 1 (all)
  x = 4 if len(data)>120 else 1
  appenditem = []
  features = []
  for i in range(len(data)):
    appenditem.append(np.mean(data[i], dtype=np.float64))
    appenditem.append(np.var(data[i], dtype=np.float64))
    appenditem.append(entropy(data[i]))
    appenditem.append(np.amax(data[i]))
    appenditem.append(np.amin(data[i]))
    #add a column when appropiate
    if(i+1) % x == 0:  
      features.append(appenditem)
      appenditem = []
  return features

def duplicate_ids(ids):
  #for each unique ID, add 2 of the same ID (1 for pain, 1 for no pain)
  id_set = sorted(set(ids))

  id_list = []
  for item in id_set:
    id_list.extend((item, item))
  return id_list

def make_features(ids, data, indexes):
  #first 2 columns are ids and label, after that add the relevant 5 feilds (mean, variance, etc.)
  columns = ["ids", "label"]
  for i in indexes:
    if i in params.keys():
      columns.extend((f'{params[i]}_mean',f'{params[i]}_variance',f'{params[i]}_entropy',f'{params[i]}_max',f'{params[i]}_min'))
  
  features = calculate_features(data)   
  id_list = duplicate_ids(ids)

  df_list = []
  # every other row should be pain vs no pain
  for i in range(0, len(id_list)):
    if i % 2 == 0:
      label = 'No Pain'
    else:
      label = 'Pain'
    df_list.append(np.array([id_list[i], label] + (features[i])))
  #create df
  df = pd.DataFrame(df_list, columns=columns)
  return df

#track time of program
start_time = time.time()
def main():
  print()
  print(f'COMMAND: {str(sys.argv)}')
  print()

  switch = {
    'dia': [0],
    'sys': [1],
    'eda': [2],
    'res': [3],
    'all': [0,1,2,3]
  }
  
  #error handling/determining what paramaters are used
  if len(sys.argv)<2 or len(sys.argv)>2:
    print("Error: Usage--python project1.py ['dia'|'eda'|'sys'|'res'|'all']")
    exit(0)
  indexes = switch.get(sys.argv[1].lower(), [])
  if len(indexes) < 1:
    print("Error: Usage--python project1.py ['dia'|'eda'|'sys'|'res'|'all']")
    exit(0)

  #preprocessing and constructing df
  ids, data = load_data(indexes)
  df = make_features(ids, data, indexes)
  #dealing with bad values
  df.replace(["inf", "-inf", "nan"], 0, inplace=True)

  # features = df[df.columns.difference(['ids', 'label'])]
  # features.to_csv('features.csv', index=False)

  # x is all colums except first 2, y is labels
  X = df.iloc[:, 2:]
  y = df["label"]
  # X.astype('float32').dtypes
  # features.astype('float32').dtypes

  #classifier and folds defined
  rf = RandomForestClassifier(max_depth=10, random_state=0)
  kFold = KFold(n_splits=10)
  
  i = 1
  accuracy = 0.0
  precision  = 0.0
  recall = 0.0
  confusion_mtrx = [[0,0],[0,0]]
  # 10 folds, no shuffle. 9 training, 1 test
  for train_index, test_index in kFold.split(X): 

    print(f"FOLD #{i}")
    i += 1

    print("TRAINING ON:")
    print(train_index)
    print("TESTING ON:")
    print(test_index)
    print()
    #setting testing and training, fitting model, and predicting
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    rf.fit(X_train,y_train)
    predict = rf.predict(X_test)
    
    # since we have 10 folds, we will sum these metrics
    accuracy += accuracy_score(y_test, predict)
    precision += precision_score(y_test, predict, pos_label='No Pain')
    recall += recall_score(y_test, predict, pos_label='No Pain')
    confusion_mtrx += confusion_matrix(y_test, predict)
 
  # then print them after dividing by 10 (for folds)
  print((f'AVERAGED METRICS OVER 10 FOLDS ON {sys.argv[1].upper()} FEATURES'))
  print(('ACCURACY: ' + str(accuracy/10)))
  print(('PRECISION: ' + str(precision/10)))
  print(('RECALL: ' + str(recall/10)))
  print(('CONFUSION MATRIX')) 
  cm = confusion_mtrx/10
 
  print("-------------")
  # pretty printing for confusion matrix!
  for i in range(2):
      cell = ""
      for j in range(2):
          cell += "| %.1f " %cm[i, j]
      cell+="|"
      cell = cell
      print(cell)        
      print(("-------------"))
  print('RUNTIME: %s' %(time.time() - start_time))      
 
if __name__ == "__main__":
  main()