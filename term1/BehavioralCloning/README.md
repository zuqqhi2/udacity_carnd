# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[train-c1]: ./images/center.png "Center Lane Driving"
[train-r1]: ./images/normal-recover1.png "Going Over The Side Lane"
[train-r2]: ./images/normal-recover2.png "Recovery"
[train-r3]: ./images/normal-recover3.png "Back To Center"
[learn-1]: ./images/learning_curve.png "Learning Curve"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 or 5x5 filter sizes and depths between 24 and 64 (model.py lines 70-74)

The model includes RELU layers to introduce nonlinearity (code lines 70-74), and the data is normalized in the model using a Keras lambda layer (code line 68).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 77, 79, 81).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 90-95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

In addition to it, there are bridge area and non lane area in the course. To make the vehicle drive on the road even if such king of situation, I used returning and recovering with sharp angle patterns.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA architecture and improve it by adding layer or tuning some hyperparameters.

My first step was to use the NVIDIA network I thought this model might be appropriate because the network is used for autonomous driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout and try some epoch numbers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at like no guard lanes and bridge to improve the driving behavior in these cases, I added some recovery patterns to imply a direction is prohibited even if there is no guard lanes.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-83) consisted of the following layers.

| Layer         		|     Description	        					               |
|:------------------|:-------------------------------------------------|
| Input         		| 160x320x3 Normalized image 							         |
| Cropping          | Cut 70 pixels from top and 25 pixels from bottom |
| Convolution 5x5   | Output channel 24, Subsample 2x2       	         |
| RELU					    |												                           |
| Convolution 5x5   | Output channel 36, Subsample 2x2       	         |
| RELU					    |												                           |
| Convolution 5x5   | Output channel 48, Subsample 2x2       	         |
| RELU					    |												                           |
| Convolution 3x3   | Output channel 64       	                       |
| RELU					    |												                           |
| Convolution 3x3   | Output channel 64       	                       |
| RELU					    |												                           |
| Flatten       		|         								                         |
| Fully connected		| outputs 100       				        					     |
| Dropout				    |	Probability 30%									                 |
| Fully connected		| outputs 50               									       |
| Dropout				    | Probability 30%							                     |
| Fully connected		| outputs 10                									     |
| Dropout				    | Probability 30%							                     |
| Fully connected		| outputs 1                									       |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. After that I also recorded two laps on track but backward. Here is an example image of center lane driving:

![Center Lane Driving][train-c1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep driving inside the lanes. These images show what a recovery looks like :

![Going Over The Side Lane][train-r1]
![Recovery][train-r2]
![Back To Center][train-r3]

To augment the data sat, I also flipped images and angles thinking that this would adding more data to the dataset and reduce data difference between left and right corner and recovery data volume and quality.


After the collection process, I had 7,806 number of data points. I then preprocessed this data by normalizing.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 3 as epoch. I tried from 2 to 10. Then highest performance one was 3 epoch in my trials. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is learning curve.
![Learning Curve][learn-1]
