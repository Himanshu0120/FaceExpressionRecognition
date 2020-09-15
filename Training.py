
import pandas as pd
import numpy as np
import keras

data=pd.read_csv('fer2013.csv')

x=data.pixels.values

for j,i in enumerate(x):
  x[j]=np.array(i.split(' ')).reshape(48, 48, 1).astype('float32')

x= np.stack(x, axis=0)

x.shape

y=data.emotion.values

#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).



import matplotlib.pyplot as plt
plt.figure(figsize=(18,10))
t=0
c0=0
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0
for i in range(1,100):
  if y[i]==0:
    c0+=1
    t+=1
    if c0>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Angry')

  if y[i]==1:
    c1+=1
    t+=1
    if c1>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Disgust')

  if y[i]==2:
    c2+=1
    t+=1
    if c2>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Fear')

  if y[i]==3:
    c3+=1
    t+=1
    if c3>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Happy')

  if y[i]==4:
    c4+=1
    t+=1
    if c4>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Sad')
  
  if y[i]==5:
    c5+=1
    t+=1
    if c5>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Surprise')

  if y[i]==6:
    c6+=1
    t+=1
    if c6>5 or t>35:
      continue
    plt.subplot(7,5,t)
    plt.imshow(x[i].reshape((48,48)),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Neutral')
  plt.tight_layout()

y= keras.utils.to_categorical(y)
y.shape

for j,i in enumerate(x):
  x[j]=i.reshape((48,48,1))

from keras.models import Sequential

from keras.layers import Dropout,Convolution2D,MaxPool2D,Dense,Flatten

model=Sequential()

model.add(Convolution2D(filters=20,kernel_size=(3,3),input_shape=(48,48,1)))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Convolution2D(filters=40,kernel_size=(3,3)))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Convolution2D(filters=80,kernel_size=(3,3)))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(150,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(rescale=1/255,horizontal_flip=True,vertical_flip=True,width_shift_range=0.2)
val_gen=ImageDataGenerator(rescale=1/255)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

train_set=train_gen.flow(x_train,y_train)
test_set=val_gen.flow(x_test,y_test)

model.info=model.fit(train_set,epochs=50,validation_data=test_set)

model.save('model.h5')

from keras.models import load_model

a=load_model('model.h5')

