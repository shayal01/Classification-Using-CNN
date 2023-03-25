# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:41:36 2022

@author: shaya
"""
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

from scipy import io

import matplotlib.pyplot as plt


    
#loading the mat file

trX=io.loadmat(r'C:\Users\shaya\Downloads\train_32x32.mat')['X']
trY = io.loadmat(r'C:\Users\shaya\Downloads\train_32x32.mat')['y']
tsX = io.loadmat(r'C:\Users\shaya\Downloads\test_32x32.mat')['X']
tsY = io.loadmat(r'C:\Users\shaya\Downloads\test_32x32.mat')['y']

# normalizing the data .

X_train=trX/255
X_test=tsX/255

#encoding the labels of both the training and test set into one hot vector
y_train = to_categorical(trY)
y_test = to_categorical(tsY)

EPOCHS=20   #no of epochs for training the model
BATCH_SIZE=3 #number of samples for updating the weights while training

input_shape = (32,32,3)

#CNN MODEL for classification

model=Sequential()
model.add(Conv2D(64,kernel_size=(5,5),  activation='relu',input_shape=input_shape,padding='same',strides=(1,1))) #covolution layer of 64 output feature maps with relu as activation function
model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))                                  #max pooling layer
model.add(Conv2D(64,kernel_size=(5,5),activation='relu',padding='same',strides=(1,1)))          #covolution layer of 64 outputs
model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))                              #max pooling layer         
model.add(Conv2D(128,kernel_size=(5,5),activation='relu',padding='same',strides=(1,1)))         #covolution layer of 128 output feature maps
model.add(Flatten())                                                           #making the input into 1 dimensional
model.add(Dense(3072, activation='relu'))   #fully connected layer   of 3072 nodes      
model.add(Dense(2048, activation='relu'))   #fully vconnected layer of 2048 nodes
model.add(Dense(10, activation='softmax'))  # fully connected layer of 10 nodes and softmax as actvation function

model.compile(loss=categorical_crossentropy,         #compiling the defined model and defining the loss function
              optimizer=SGD(learning_rate=.01),     #optimizer and metrics 
              metrics=['accuracy'])
hist=model.fit(X_train, y_train,                      #training the model on the train set
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          )

#plotting accuracy of train  and test sets vs number of epochs
a=plt.figure(1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
a.show()
#plotting loss of train and test sets vs number of epochs
b=plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
b.show()


score = model.evaluate(X_test, y_test, verbose=0) #testing the model on test set
print('Test loss:', score[0]) #prediction accuracy
print('Test accuracy:',score[1]) #prediction loss


