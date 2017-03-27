#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a NVIDIA Behavioral Cloning Architecture with a small variation, I added a dropout and I changed the number of neurons in the fully connected layers.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81). 

I also made a set of 100,000 images to prevent overfitting with a training in only 5 epochs.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, I also used images from the second circuit.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train only a few times over a convnet with a lot of data.

My first step was to use a convolution neural network model similar to the NVIDIA Behavioral Cloning Architecture I thought this model might be appropriate because is small and powerfull.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added a Dropout and I reduced the number of epochs.

Then I took more data from the circuits.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I just collected more data

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 69 - 90) consisted of a convolution neural network with the following layers and layer sizes:

1- Convolution with an output depth of 24, with a subsample of 2 by 2 and activation with relu.
2- Convolution with an output depth of 36, with a subsample of 2 by 2 and activation with relu.
3- Convolution with an output depth of 48, with a subsample of 2 by 2 and activation with relu.
4- Convolution with an output depth of 64, with a subsample of 2 by 2 and activation with relu.
5- Convolution with an output depth of 64, with a subsample of 2 by 2 and activation with relu.

6- Dropout of 0.5

7- Flatten layer

8- Fully Connected layer of 128 neurons
9- Fully Connected layer of 64 neurons
10- Fully Connected layer of 10 neurons
11- Fully Connected layer of 1 neuron to get the steering angle

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the track if it goes out. 

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would improve the steering angle in the right curves.

After the collection process, I had about 100,000 number of data points. I then preprocessed this data by normalization and cropping it to remove the data I don't needed.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
