import csv

lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

import matplotlib.pyplot as plt
import cv2
import sklearn
from sklearn.utils import shuffle
import numpy as np

path='./data/IMG/'
correction=0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while True: # Loop forever so the generator never terminates
        shuffle(samples)        
            
        for offset in range(0, num_samples, batch_size):            
            batch_samples = samples[offset:offset+batch_size]
            images=[]
            measurements=[]
            flipped_images=[]
            angles=[]
            
            for batch_sample in batch_samples:
              for i in range(3):
                source_path=batch_sample[i]
                filename=source_path.split('\\')[-1]
                local_path=path+filename
                image = plt.imread(local_path)
                images.append(image)                
              measurement = float(batch_sample[3])
              measurements.append(measurement)
              measurements.append(measurement+correction)
              measurements.append(measurement-correction)
              
              for image, measurement in zip(images,measurements):
                flipped_images.append(image)
                flipped_images.append(cv2.flip(image,1))
                angles.append(measurement)
                angles.append(measurement * -1.0)
              
            # trim image to only see section with road
            X_train = np.array(flipped_images)
            y_train = np.array(angles)
            #X_train = np.resize(X_train,(32,160,320,3))
            #y_train = np.resize(y_train, (32,1))
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.1))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples) / 32),\
                    validation_data=validation_generator, validation_steps=int(len(validation_samples) / 32),\
                    epochs=3, verbose=2)

model.save('model.h5')