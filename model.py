import os
import csv

samples = [] 
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Split the data into validation and train sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import cv2
import numpy as np
import sklearn
from random import shuffle


def generator(samples, batch_size=32):
    correction = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #For each row we take the right camera image, the left and the center ones
            for batch_sample in batch_samples:
                name = batch_sample[0]
                name_left = batch_sample[1]
                name_right = batch_sample[2]
                center_image = cv2.imread(name)
                img_left = cv2.imread(name_left)
                img_right = cv2.imread(name_right)
                center_angle = float(batch_sample[3])
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                images.append(center_image)
                images.append(img_left)
                images.append(img_right)
                angles.append(center_angle)
                angles.append(steering_left)
                angles.append(steering_right)

           #Augmentate the data by flipping the images
            augmented_images, augmented_angles = [],[]
            for image,measurement in zip(images,angles):
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(measurement*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 90, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
#Normalize the data
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320,3)))
#Cropping the image
model.add(Cropping2D(cropping=((70,25),(0,0))))
"""NVIDIA Behavioral cloning network variation"""
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=7, verbose=1)

model.save('model2.h5')
print("model saved!")
#exit()

import matplotlib.pyplot as plt
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
