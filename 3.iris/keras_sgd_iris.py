from keras.models import Sequential      
from keras.layers.core import Dense, Activation    
from keras.optimizers import SGD  
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import numpy as np   
import pandas as pd

#input data  
def get_iris_inputs(filename):
    dataframe = pd.read_csv(filename, header=None)
    dataset = dataframe.values
    x_train = dataset[:, 0:4].astype(float)
    y_label = dataset[:, 4].astype(float)

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y_label)
    train_labels = np_utils.to_categorical(encoded_Y, 3)
    return x_train,train_labels

x_train,train_labels = get_iris_inputs("iris_trainingk.csv")
print(x_train)
print(train_labels)
#create models, with 1hidden layers      
model = Sequential()      
model.add(Dense(10, input_dim=4, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

#training  
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])  
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])     
hist = model.fit(x_train, train_labels, batch_size=1, epochs=1000, shuffle=True, verbose=2,validation_split=0.0)  
print(hist.history)  
#evaluating model performance  
test_train,test_label = get_iris_inputs("iris_testk.csv")
loss_metrics = model.evaluate(test_train, test_label, batch_size=1) 
print(loss_metrics)
#print(model.predict(x_train).round())
