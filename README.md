# **Behavioral Cloning** 

---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-cnn-architecture.jpg "Model Visualization"
[image2]: ./examples/center_image.jpg "Center Image"
[image3]: ./examples/recovery_center_1.jpg "Recovery Image"
[image4]: ./examples/recovery_center_2.jpg "Recovery Image"
[image5]: ./examples/recovery_center_3.jpg "Recovery Image"
[image6]: ./examples/orig_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 82-101) 

The model includes RELU activation function to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (model.py line 83) and images are cropped (to see only road section) in the model using Cropping2D layer (model.py line 84).

#### 2. Attempts to reduce overfitting in the model

The model contains multiple dropout layers; one after all convolution layers and others in between each fully connected layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 107-109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving of two laps, recovering from the left and right sides of different types of the road and one lap of counter-clockwise driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture is to train on different road images generated by simulator and learn corresponding sterring angle respective to the position of the car on the road.

My first step was to use a convolution neural network model similar to the NVIDIA deep learning architecture. I thought this model might be appropriate because this is the proven, simple and most efficient model used by NVIDIA for their self driving cars deep learning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I eventually added more data by driving one more lap, then added data for counter clockwise driving and then finally added dropout layers in between each fully connected layer so that the model will not overfit on the training data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I have collected recovery data to recover from the edges of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-101) consisted of a convolution neural network with the following layers and layer sizes | Layer         | Layer Specs                                | Output Size |
|---------------|--------------------------------------------|-------------|
| Normalization | lambda x: x / 255 - 0.5                    | 160x320x3   |
| Cropping      | Cropping2D(cropping=((70, 25), (0, 0))     | 65x320x3    |
| Convolution   | 24, 5x5 kernels, 2x2 subsampling           | 31x158x24   |
| RELU          | Non-linearity                              | 31x158x24   |
| Convolution   | 36, 5x5 kernels, 2x2 subsampling           | 14x77x36    |
| RELU          | Non-linearity                              | 14x77x36    |
| Convolution   | 48, 5x5 kernels, 2x2 subsampling           | 5x37x48     |
| RELU          | Non-linearity                              | 5x37x48     |
| Convolution   | 64, 3x3 kernels, 1x1 subsampling           | 3x18x64     |
| RELU          | Non-linearity                              | 3x18x64     |
| Convolution   | 64, 3x3 kernels, 1x1 subsampling           | 1x16x64     |
| RELU          | Non-linearity                              | 1x16x64     |
| Dropout       | Probabilistic regularization (p=0.1)       | 1x16x64     |
| Flatten       | Convert to vector                          | 1024        |
| Dense         | Fully connected layer                      | 100         |
| Dropout       | Probabilistic regularization (p=0.1)       | 100         |
| Dense         | Fully connected layer                      | 50          |
| Dropout       | Probabilistic regularization (p=0.1)       | 50          |
| Dense         | Fully connected layer                      | 10          |
| Dense         | Output prediction layer                    | 1           |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it approaches the left or right edges of the road. These images show what a recovery looks like starting from top to the bottom:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded one lap of driving in counter clockwise direction in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the model learning instead of memorizing the path. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 15253 number of data points. I then randomly shuffled the data set and put 20% of the data into a validation set. I then passed this data to generator which preprocessed this data by normalizing and trimming the images by 70 pixels of top and 25 pixels of botoom of the image so that it only contains roads and not any other objects, and then produces the images for training and validating in batches instead of storing entire data in memory.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation loss pleatuing after 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.