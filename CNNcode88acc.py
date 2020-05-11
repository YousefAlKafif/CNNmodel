# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:00:43 2019

@author: YOUSEF AL-KAFIF
"""

# Using the Keras API, running on TensorFlow backend.
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os
import h5py # Needed to save keras model into a HDF5 file which saves its architecture, weights, training configurations, and the state of the optimizer.
     
class LossHistory(Callback): # Creating a loss history class to log the models history.
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
     
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
         
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
  

# PART 1 - BUILDING THE CNN

# Initialising the CNN model :
classifier = Sequential()
     
# Step 1 - Convolution :
input_size = (128, 128) #Input size of images is 128 height, 128 width.
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu')) #Using 32 filters/feature detectors that are 3x3 matrices which will create 32 feature maps. 
                                                                                    #Then passes through the rectifier 'relu' activation function in order to eliminate negative values, and reduce linearity. 
# Step 2 - Pooling :
classifier.add(MaxPooling2D(pool_size=(2, 2)))  #Max pooling with a 2x2 matrice, i.e. recording the highest values of the feature map and creating a new pooled feature map from those values.
                                              
# Adding a 2nd convolutional + Max Pooling layer :
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
     
# Adding a 3rd convolutional + Max Pooling layer :
classifier.add(Conv2D(64, (3, 3), activation='relu')) #Doubling the amount of filters/feature detectors.
classifier.add(MaxPooling2D(pool_size=(2, 2)))
     
# Step 3 - Flattening :
classifier.add(Flatten()) # 'Flattening' the feature maps values into a vector which will be the input to the fully connected ANN layers.
     
# Step 4 - Fully connected ANN layer :
classifier.add(Dense(units=64, activation='relu'))  #Using 64 nodes/neurons
classifier.add(Dropout(0.5)) #Dropout rate of 50%. i.e. disabling 50% of the nodes at each update during training.
classifier.add(Dense(units=1, activation='sigmoid')) #The output layer using the Sigmoid activation function as is common practice when predicting probabilities.
     
# Compiling the CNN :
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #Using the 'Adam' gradient descent algorithm as my optimizer, and using binary_crossentropy as this a two class problem (cat or dog)
  

   
# PART 2 - FITTING THE CNN TO THE IMAGES. i.e. training the model.

batch_size = 32 #will update after every 32 random images. i.e. after every batch.

#Scaling and data augmenting my training set :
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
#Scaling my test set :     
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Importing my training set images :     
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=input_size,
                                                batch_size=batch_size,
                                                class_mode='binary')
#Importing my test set images :     
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')
     
# Creating a loss history class object :
history = LossHistory()

# Fitting the model i.e. training it :     
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size, #Amount of batches to be completed before declaring an epoch to be finished.
                         epochs=90, 
                         validation_data=test_set, 
                         validation_steps=2000/batch_size,
                         workers=12,  #adjusted workers and maxQsize for my personal GPU performance.
                         max_queue_size=100,
                         callbacks=[history]) #recording training stats into history class object.
     


# PART 3 - MAKING PREDICTIONS, SAVING MODEL, SAVING LOSS HISTORY TO FILE.

# Saving model :
model_path = 'dataset/cat_or_dog_model.h5'
classifier.save(model_path)
print("Model saved to", model_path)
     
# Saving loss history to file : 
lossLog_path = 'dataset/loss_history.log'
myFile = open(lossLog_path, 'w+')
myFile.write(history.losses)
myFile.close()

# Clearing session :
backend.clear_session()

# Confirming class indices:
print("The model class indices are:", training_set.class_indices)

# Predicting a new image :
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = input_size)
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis = 0) #Adding an extra dimension to the image as the .predict() function expects a 4D array w/ the 4th dimension corresponding to the batch.
result = classifier.predict(test_image) 

if result[0][0] == 1: # The .predict() function returns a 2D array, thus [0][0] corresponds to the first element.
    prediction = 'dog'
else:
    prediction = 'cat'
        
print ("This is a",prediction)

