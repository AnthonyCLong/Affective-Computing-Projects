import os
import random
import cv2
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def listFiles(file_path, file_ext):
    return [os.path.join(path, file)
        for path, _, files in os.walk(file_path)
            for file in files if file.endswith(file_ext)]

def processData(height, width, path, ext ):
    files = listFiles(f'{path}\{ext}', '.jpg')
    
    labels = []
    for i in files:
        try:
            
            path = i.split("\\")
            label = path[-2]
            print("Processing " + path[-3] + " file: " + path[-1])
            gray_img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)

            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray_img, 1.5, 4)
            
            for (x, y, w, h) in faces:
                bounded_img = gray_img[int(y):int(h), int(x):int(w)]
                intrest_region = cv2.resize(bounded_img, (width, height), 1)
                data = np.asarray(intrest_region).astype('float')
                labels.append([data, label])
                
        except:
            continue
    
    return labels

def constructModel(shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(.33))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(.33))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

batches = 32
epochs = 20

X_train, y_train, X_validate, y_validate, X_test, y_test = ([] for i in range(6))

if (len(sys.argv) != 4):
    print("USAGE: python cnn.py width height \"Abs_dir_with_quotes\"")
    exit(1)

height = int(sys.argv[1])
width = int(sys.argv[2])
path = sys.argv[3]

shape = (width, height, 1)

train_set = processData(height, width, path, "Training")
valid_set = processData(height, width, path, "Validation")
test_set  = processData(height, width, path, "Testing")

random.shuffle(train_set)

for features, labels in train_set:
    X_train.append(features)
    y_train.append(labels)

X_train = np.array(X_train).reshape(-1, width, height, 1)
X_train /= 255.0

for features, labels in valid_set:
    X_validate.append(features)
    y_validate.append(labels)

X_validate = np.array(X_validate).reshape(-1, width, height, 1)
X_validate /= 255.0

for features, labels in test_set:
    X_test.append(features)
    y_test.append(0 if labels == 'No_pain' else 1)

X_test = np.array(X_test).reshape(-1, width, height, 1)
X_test /= 255.0

y_train = pd.get_dummies(y_train)
y_validate = pd.get_dummies(y_validate)
y_test_encode = pd.get_dummies(y_test)

adam_model = constructModel(shape)
adam_model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=epochs, batch_size=batches)
adam_results = adam_model.evaluate(X_test, y_test_encode, verbose=0)

y_predict = adam_model.predict(X_test)
y_predict = y_predict.argmax(axis=1)
matrix = confusion_matrix(y_test, y_predict)

print("ACCURACY: " + str(adam_results[1]))
print("LOSS: " + str(adam_results[0]))
print("PRECISION SCORE: " + str(precision_score(y_test, y_predict)))
print("RECALL SCORE: " + str(recall_score(y_test, y_predict)))
print("F1 SCORE: " + str(f1_score(y_test, y_predict)))
print(matrix)