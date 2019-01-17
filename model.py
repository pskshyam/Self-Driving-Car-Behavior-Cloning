# Import necessary libraries
import cv2
import csv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

# Read the driving log csv generated in simulator
# and store the lines in a list
lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#print(len(lines))

# Split the lines data into train and validation split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
#print(len(train_samples))
#print(len(validation_samples))

path='./data/IMG/' # Path where images are placed for training
correction=0.2 # Correction factor for left and right camera images
batch_size=32 # Batch size for the generator

# Define a generator function
def generator(samples, batch_size=batch_size):
	'''This function takes in input of images data and batch size to run the generator on.
	For every batch,
		It reads in images and measurements data from the csv file and stores them in two separate lists.
		It adds and subtracts correction factor to each measurement for left and right camera images.
		It flips images and reverses their corresponding angles and stores them in two separate lists.
		Finally it converts the lists into arrays and yields them in X_train and y_train for that batch of samples.
	'''
    num_samples = len(samples)
    
    while True: # Loop forever so the generator never terminates
        shuffle(samples)        
            
        for offset in range(0, num_samples, batch_size):       	# Loop through each batch     
            batch_samples = samples[offset:offset+batch_size]  
			
			#Define placeholders for images and their steering angles
            images=[]
            measurements=[]
            flipped_images=[]
            angles=[]
            
            for batch_sample in batch_samples: 				   		# Loop through each sample in the batch
              for i in range(3):							   		# Loop through first three columns of the csv file	
                source_path=batch_sample[i]	
                filename=source_path.split('\\')[-1]				# Extract the file name from source path
                local_path=path+filename							# Merge the file name with local path
                image = plt.imread(local_path)						# Read the image from the local path
                images.append(image)                	
              measurement = float(batch_sample[3])					# Read the measurement from csv file
              measurements.append(measurement)	
              measurements.append(measurement+correction)			# Add correction factor for the right camera images
              measurements.append(measurement-correction)			# Subtract correction factor for the left camera images
              
              for image, measurement in zip(images,measurements):	# Loop through the images and measurements 
                flipped_images.append(image)
                flipped_images.append(cv2.flip(image,1)) 			# Flip the image horizontally
                angles.append(measurement)
                angles.append(measurement * -1.0)              		# Reverse the steering angle
            
            X_train = np.array(flipped_images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Call generator function for training and validation samples
train_generator = generator(train_samples, batch_size=batch_size)	
validation_generator = generator(validation_samples, batch_size=batch_size)

# NVIDIA Model Architecture
# 5 Convolution2D layers followed by 4 Fully connected layers
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))	# Normalize images
model.add(Cropping2D(cropping=((70,25),(0,0)))) 					# Trim image to only see section with road
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

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the generator data to the model
model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples) / batch_size),\
                    validation_data=validation_generator, validation_steps=int(len(validation_samples) / batch_size),\
                    epochs=3, verbose=2)

# Save the model
model.save('model.h5')