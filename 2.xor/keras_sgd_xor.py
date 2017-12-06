from keras.models import Sequential      
from keras.layers.core import Dense, Activation    
from keras.optimizers import SGD  
import numpy as np   
  
#input data  
x_train =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
y_label = np.array([0,1,1,0])  
  
#create models, with 1hidden layers      
model = Sequential()      
model.add(Dense(16, input_dim=2, activation='sigmoid'))
#model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#training  
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])  
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['binary_accuracy'])     
hist = model.fit(x_train, y_label, batch_size=1, nb_epoch=1000, shuffle=True, verbose=0,validation_split=0.0)  
print(hist.history)  
#evaluating model performance  
#loss_metrics = model.evaluate(x_train, y_label, batch_size=1) 
#print(loss_metrics)
print(model.predict(x_train).round())
