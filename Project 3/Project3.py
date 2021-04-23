from operator import index
import time
import sys
import csv
import numpy as np
from numpy import testing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import minmax_scale
from statistics import mode

# read in data. if the rows value is in index list (meaning we want it), append appropiate things
def load_data(filename):
  ids = []
  data = []
  with open(filename, 'r') as file:
    reader = csv.reader(file)
    for i,row in enumerate(reader):          
      ids.append(row[0])
      raw = np.array(row[3::]).astype(np.float)
      scaled_normalized = minmax_scale(down_sample(raw))
      data.append(scaled_normalized)
      #fomatting to know were not stalling
      # if (i+1) % 20 == 0:
      #   print('READING COLUMNS',i-19,'-',i)
  # print()
  return ids, data

def split_features(data):
  # print("CRAFTING FEATURES...")
  # print()
  dia = []
  eda = []
  sys = []
  res = []
  for i in range(len(data)):
    if i % 4 == 0: 
      dia.append(data[i])
    elif i % 4 == 1: 
      eda.append(data[i])
    elif i % 4 == 2: 
      sys.append(data[i])
    elif i % 4 == 3: 
      res.append(data[i])
  return dia, eda, sys, res

def duplicate_ids(ids):
  #for each unique ID, add 2 of the same ID (1 for pain, 1 for no pain)
  id_set = sorted(set(ids))
  id_list = []
  for item in id_set:
    id_list.extend((item, item))
  return id_list

def make_features(ids, data):
  #first 2 columns are ids and label, after that add the downscaled data
  columns = ["ids", "label"]
  for i in range(0,5000):
    columns.append(f'{i}')
  
  dia, eda, sys, res = split_features(data)   
  dia_df = [] 
  eda_df = [] 
  sys_df = [] 
  res_df = [] 

  dataframes = [dia_df, eda_df, sys_df, res_df]
  datatypes = [dia, eda, sys, res]
  id_list = duplicate_ids(ids)

  for i, data in enumerate(datatypes):
    df_list = []
    # every other row should be pain vs no pain
    for j in range(0, len(id_list)):
      if j % 2 == 0:
        label = 'No Pain'
      else:
        label = 'Pain'
      df_list.append(np.array([id_list[j], label] + (data[j]).tolist()))
    #create df
    dataframes[i] = pd.DataFrame(df_list, columns=columns)
  return dataframes

def down_sample(data):
  scaled = []
  frames = len(data)
  #calculated ratio
  ratio = int(frames/5000)
  
  #for each grouping in ratio, average the data into one of the 5000 values
  for ind in range(0, frames, ratio):
      scaled.append(sum(data[ind:ind+ratio])/ratio)
  #note: since ratio will not scale perfectly, we must cut off the extras after downscaling
  return scaled[0:5000]

def printCM(cm):
   # pretty printing for confusion matrix!
  print("-------------")
  for i in range(2):
      cell = ""
      for j in range(2):
          cell += "| %.1f " %cm[i, j]
      cell+="|"
      cell = cell
      print(cell)        
      print(("-------------"))

def main():
  # print()
  # print(f'COMMAND: {str(sys.argv)}')
  # print()

  
  #error handling/determining what paramaters are used
  if len(sys.argv) != 3:
    print("Error: Usage--python project3.py 'file1' 'file2'")
    exit(0)

  #preprocessing and constructing df
  types = ['DIA', 'EDA', 'SYS', 'RES']
  train_ids, train_data = load_data(sys.argv[1])
  test_ids, test_data = load_data(sys.argv[2])

  training_data = make_features(train_ids, train_data)
  testing_data = make_features(test_ids, test_data)

  forest = [RandomForestClassifier(max_depth=100, random_state=0) for i in range(4)]
  kFold = KFold(n_splits=3)
  
  predictions = []
  for i in range(0, 60):
        predictions.append([])
  
  for ind in range(4):
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    confusion_mtrx = [[0, 0], [0, 0]]

    tree = forest[ind]
    train = training_data[ind]
    test = testing_data[ind]
    dType = types[ind]

    features = train[train.columns.difference(['ids', 'label'])]
    X = features
    y = train["label"]
    for train_index, validate_index in kFold.split(X):
        # print("TRAINING ON:")
        # print(train_index)
        # print("TESTING ON:")
        # print(validate_index)
        # print()

        X_train = X.iloc[train_index]
        X_validate = X.iloc[validate_index] 
        y_train = y.iloc[train_index]
        y_validate = y.iloc[validate_index]

        tree.fit(X_train, y_train)
        prediction = tree.predict(X_validate)
        
        accuracy += accuracy_score(y_validate, prediction)
        precision += precision_score(y_validate, prediction, pos_label='No Pain', zero_division=1)
        recall += recall_score(y_validate, prediction, pos_label='No Pain')
        confusion_mtrx += confusion_matrix(y_validate, prediction)

    # print((f'AVERAGED METRICS OVER 3 FOLDS ON {dType} FEATURES'))
    # print(('ACCURACY: ' + str(accuracy/3)))
    # print(('PRECISION: ' + str(precision/3)))
    # print(('RECALL: ' + str(recall/3)))
    # print(('CONFUSION MATRIX')) 
    # printCM(confusion_mtrx/3)
    
    features = test[test.columns.difference(['ids', 'label'])]
    X_test = features
    y_test = test["label"]

    prediction = tree.predict(X_test)
    for i, j in enumerate(prediction):
        predictions[i].append(j)
    
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, pos_label='No Pain', zero_division=1)
    recall = recall_score(y_test, prediction, pos_label='No Pain')
    confusion_mtrx = confusion_matrix(y_test, prediction)

    # print((f'AVERAGED METRICS FOR {dType} OVER {sys.argv[2]} TESTING FEATURES'))
    # print('ACCURACY: ' + str(accuracy))
    # print('PRECISION: ' + str(precision))
    # print('RECALL: ' + str(recall))
    # print('CONFUSION MATRIX')
    # printCM(confusion_mtrx)

    majorities = []
    for i in predictions:
      majorities.append(mode(i))

  #WHAT IS THIS??
  y_test = testing_data[0]["label"]

  accuracy = accuracy_score(y_test, majorities)
  precision = precision_score(y_test, majorities, pos_label='No Pain', zero_division=1)
  recall = recall_score(y_test, majorities, pos_label='No Pain')
  confusion_mtrx = confusion_matrix(y_test, majorities)

  print((f'METRICS FOR MAJORITY VOTING ON {sys.argv[2]} TESTING FEATURES'))
  print('ACCURACY: ' + str(accuracy))
  # print('PRECISION: ' + str(precision))
  # print('RECALL: ' + str(recall))
  # print('CONFUSION MATRIX')
  # printCM(confusion_mtrx)

if __name__ == "__main__":
  main()