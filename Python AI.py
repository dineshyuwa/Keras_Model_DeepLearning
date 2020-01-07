import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import SGD

import keras

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

RESHAPED=784
x_train=x_train.reshape(60000,RESHAPED)
x_test=x_test.reshape(10000,RESHAPED)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train/=255
x_test/=255

x_train.shape
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

model=Sequential()
model.add(Dense(10,input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=200,verbose=1,validation_split=0.2)
model.evaluate(x_test,y_test)