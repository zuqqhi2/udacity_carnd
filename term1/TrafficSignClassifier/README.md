# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[inv-1]: ./images/traindata-6-sample.jpg "Random 6 sample images from training data"
[inv-2]: ./images/label-number-dist.jpg "Item number for each labels in each dataset"
[pre-1]: ./images/pre-grayscale.png "Grayscale"
[pre-2]: ./images/pre-equalized-histogram.png "Histogram Equalization"
[pre-3]: ./images/pre-centralized.png "Centralization"
[new-1]: ./images/30km.jpg "Traffic Sign 1"
[new-2]: ./images/70km.jpg "Traffic Sign 2"
[new-3]: ./images/aller-art.jpg "Traffic Sign 3"
[new-4]: ./images/no-entry.jpg "Traffic Sign 4"
[new-5]: ./images/stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

First is random sampled images from training data set.
Looks brightness is not normalized.

![Random 6 sample images from training data][inv-1]

Next is histogram of each label data number.
Looks each label has same number of data.

![Item number for each labels in each dataset][inv-2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because we can see type of traffic sign from only shape.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][pre-1]

As a next step, I equalized histogram to improve image's contrast.

Here is an example of a traffic sign image before and after histogram equalization.

![Histogram Equalization][pre-2]

As a last step, I normalized the image data because brightness difference can affects learning result.

Here is an example of a traffic sign image before and after normalization.

![Normalization][pre-3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					          |
|:------------------|:-------------------------------------------|
| Input         		| 32x32x1 Grayscale image   							    |
| Convolution 3x3   | 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					    |												                      |
| Max pooling	      | 2x2 stride, outputs 16x16x32 				        |
| Convolution 3x3   | 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					    |												                      |
| Max pooling	      | 2x2 stride, outputs 8x8x64  				        |
| Convolution 3x3   | 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					    |												                      |
| Convolution 3x3   | 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					    |												                      |
| Max pooling	      | 2x2 stride, outputs 4x4x128 				        |
| Flatten       		| outputs 2048        								        |
| Fully connected		| outputs 400       				        					|
| RELU					    |												                      |
| Dropout				    |	Probability is 50%									        |
| Fully connected		| outputs 200               									|
| RELU					    |												                      |
| Dropout				    | Probability is 50%							            |
| Fully connected		| outputs 43                									|
| Softmax				    | Network output                          		|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer because Adam is standard optimizer.
Actually I didn't compare other optimizers but I could achieve expected accuracy.

Followings are hyperparameters which I used.
Actually there are no theoretical reason for the values.
I just tried some patterns and pickup best performance one.

| Parameter			   | Value	            |
|:-----------------|:-------------------|
| Epoch      		   | 50   	            |
| Batch size   	   | 128 		            |
| Learning Rate	   | 0.001	            |
| LeNet Parameters | See previous table |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.937
* test set accuracy of 0.924


I used iterative approach to improve accuracy of the network.
My first model is just normal LeNet which I studied in the course because LeNet performance is good for image classification.

I tried some smaller learning rate with first net at first.
But I couldn't get good result than I expected.

So, I thought there were two possible problems in the net. The one is that there were many local minimum before 0.93 and it's difficult to escape from local minimum, the another is model doesn't have enough ability to classifier traffic signs.

I tried to adding more layer and channels after that.
Complex network is easy to do overfitting, so I tried to add dropout to avoid it.

After adding layer, channels and dropout, I could achieve the target accuracy with only tuning epoch.
Why I increase epoch is that new net is more complex than first model and it need more time to converge.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new-1]
![alt text][new-2]
![alt text][new-3]
![alt text][new-4]
![alt text][new-5]

The second, third and fifth images might be difficult to classify because traffic sign is small in the image and traffic sign is rotated.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			           | Prediction	        					|
|:---------------------|:-----------------------------|
| Vehicle traffic ban  | Keep left            				|
| 30 km/h              | 30 km/h 										  |
| No Entry		         | No Entry											|
| Stop sign	           | 50 km/h   					 				  |
| 70 km/h			         | Traffic Signal     					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy is quite low compared with test set.

But this is kind of same as my expectation because training, validation and test set images are only not rotated, image at the center.
In this case the network is basically difficult to learn about rotated image and no object at the center case.
And also resizing collapsed the image and make it difficult to detect.
"No Entry" and "30 km/h" images from web are kind of same situation of dataset images. That's why my net can detect correctly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 4th cell from Step3 title of the Ipython notebook.

For the first image, the model is completely not sure the image is because even if 1st one probability is 0.172. Actually there is no correct answer in top 5. As I mentioned, this image's situation is different from training set.  So, my model should not good at this kind of image. The top five soft max probabilities were

| Probability   |     Prediction	        			|
|:--------------|:------------------------------|
| .172         	| Keep left             				|
| .133     		  | Road works 								    |
| .126					| Ahead only									  |
| .075	      	| Roundabout  				 				  |
| .055				  | Wild animals      						|


For the second image, the model is completely sure that this is a 30 km/h sign (probability of 1.0), and the image does contain a 30 km/h sign. The top five soft max probabilities were

| Probability   |     Prediction	        			|
|:--------------|:------------------------------|
| 1.0         	| 30 km/h               				|
| .0      		  | 50 km/h   								    |
| .0	 	 			  | Passing limits							  |
| .0 	        	| 70 km/h  				    				  |
| .0			  	  | 20 km/h           						|

For the third image, the model is completely sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability   |     Prediction	        			|
|:--------------|:------------------------------|
| 1.0         	| No entry               				|
| .0      		  | Priority road							    |
| .0	 	 			  | Turn left ahead							  |
| .0 	        	| Turn right ahead    				  |
| .0			  	  | 30 km/h           						|

For the forth image, the model is completely sure that this is a give way sign (probability of 0.951), but the image does contain a stop sign. The fifth candidate is stop sign but probability of 0.004. I think the reason is training data set doesn't contain rotated image so the model doesn't care rotation as I mentioned. The top five soft max probabilities were

| Probability   |     Prediction	        			|
|:--------------|:------------------------------|
| .951         	| Give way               				|
| .016     		  | Keep right  							    |
| .012	 	 		  | Keep left     							  |
| .005 	       	| Turn left ahead 	   				  |
| .004		   	  | Stop sign          						|

For the fifth image, the model is completely not sure the image is because even if 1st one probability is 0.240. Actually there is no correct answer in top 5. As I mentioned, this image's situation is different from training set. Especially original image is too big to resize. The top five soft max probabilities were

| Probability   |     Prediction	        			|
|:--------------|:------------------------------|
| .240         	| Traffic signals        				|
| .148     		  | General caution   				    |
| .132	 	 		  | Road narrows		  					  |
| .075 	       	| 120 km/h  			    				  |
| .055			 	  | Bicycles           						|



For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
