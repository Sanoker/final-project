from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

import csv

import numpy as np
import os
import pandas as pd
import glob
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import callbacks
import time


from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 

# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))



# define function to load train, test, and validation datasets
def load_dataset(path,tar):
    files=[]
    for filename in os.listdir(path):
        #print(filename)
        files.append(os.path.join(path,filename))
        
    targets = np_utils.to_categorical(np.array(tar))
    #print(targets)
    #targets = to_categorical(np.array(data['target']), tar)
    return files, targets



df = pd.read_csv('final.csv')

csvFile=np.array(df)

x=csvFile[:,0]
targetFM=x[~np.isnan(x)]/1000
x=csvFile[:,1]
targetFMM=x[~np.isnan(x)]/1000
x=csvFile[:,2]
targetFS=x[~np.isnan(x)]/1000

x=csvFile[:,3]
targetRM=x[~np.isnan(x)]/1000
x=csvFile[:,4]
targetRMM=x[~np.isnan(x)]/1000
x=csvFile[:,5]
targetRS=x[~np.isnan(x)]/1000

x=csvFile[:,6]
targetSM=x[~np.isnan(x)]/1000
x=csvFile[:,7]
targetSMM=x[~np.isnan(x)]/1000
x=csvFile[:,8]
targetSS=x[~np.isnan(x)]/1000





path='./dataset/frontminor/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetFM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/FM.h5')



path='./dataset/frontmoderate/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetFMM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/FMM.h5')






path='./dataset/frontsevere/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetFS)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/FS.h5')














path='./dataset/Rearminor/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetRM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/RM.h5')



path='./dataset/Rearmoderate/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetRMM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/RMM.h5')


path='./dataset/Rearsevere/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetRS)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(31, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/RS.h5')










path='./dataset/sideminor/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetSM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(41, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/SM.h5')



path='./dataset/sidemoderate/'
# load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetSMM)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
########
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/SMM.h5')


path='./dataset/sidesevere/'
#load train, test, and validation datasets
train_files, train_targets = load_dataset(path,targetSS)

test_files=train_files
test_targets = train_targets
for file in train_files: assert('.DS_Store' not in file)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

img_width, img_height = 224, 224
batch_size = 4
epoch=20
######
img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
lr = 0.0004

input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(51, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=10)
model.save('Models/SS.h5')
