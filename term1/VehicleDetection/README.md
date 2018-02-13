**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[dataset]: ./output_images/car_noncar_image_sample.png
[hogv]: ./output_images/hog-vehicle.png
[hogn]: ./output_images/hog-non-vehicle.png
[pipeline]: ./output_images/pipeline.png
[heatmap0]: ./output_images/heatmap0.png
[heatmap1]: ./output_images/heatmap1.png
[heatmap2]: ./output_images/heatmap2.png
[heatmap3]: ./output_images/heatmap3.png
[heatmap4]: ./output_images/heatmap4.png
[heatmap5]: ./output_images/heatmap5.png
[final0]: ./output_images/final0.png
[final1]: ./output_images/final1.png
[final2]: ./output_images/final2.png
[final3]: ./output_images/final3.png
[final4]: ./output_images/final4.png
[final5]: ./output_images/final5.png
[video]: ./output_images/project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


At first the code loads `vehicle` and `non-vehicle` images (`Vehicle and Non-vehicle Images` term).

Here is an example of sample images of vehicle and non-vehicle:

![Car and Not Car Image Sample][dataset]

Then I tried different color space and `skimage.hog()` parameters like `orientations`, `pixels_per_cell`, and `cells_per_block`. After that the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` were best.

Here is an example of HOG result of 2 classes:

![Vehicle HOG Example][hogv]

![Non Vehicle HOG Example][hogn]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally I decided to use following(same as I showed example images at previous section):

* color space : YUV
* orientations : 9
* pixels_per_cell : (8, 8)
* cells_per_block : (2, 2)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Random Forest Classifier using HOG, color histogram and spatial features.

At first I calculated HOG features from vehicle and non vehicle image set with all channel of YCrCb color space. Then I shuffled it to avoid overfitting with order and split feature data to training data and test data.

I used `StandardScaler` to normalize features. And I fed it to the classifier. To find best hyper parameters, I used `GridSearchCV`. The best parameters are following:

* num estimators: 40
* max depth: 7
* min samples split: 4

The trained model had accuracy of 96.09% on test dataset.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search with fixed three set of y axis start, end value and scale by trying some variation.

1. ystart = 360, ystop=560, scale=1.5
2. ystart = 400, ystop=600, scale=1.8
3. ystart = 440, ystop=700, scale=2.5

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My pipeline does following 3 steps:

1. Preprocessing
2. Finding car positions including false positives
3. Get rid of false positives

At step 2, the pipeline extracts only HOG features with YUV color space. Here is an example image of the pipeline:

![Pipeline Result Sample][pipeline]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I searched windows in each frame of the video and get positive detections from my classifier. Then I created a heatmap and find correct vehicle positions with a certain threshold. I used `scipy.ndimage.measurements.label()` to identify vehicle positions in the heatmap. After that I generated bounding boxes to cover the vehicle positions. To improve robustness I use some old frame's heatmaps.

Here's an example result showing the heatmap, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:

### Here are six frames and their corresponding heatmaps:

![Heatmap Sample 0][heatmap0]
![Heatmap Sample 1][heatmap1]
![Heatmap Sample 2][heatmap2]
![Heatmap Sample 3][heatmap3]
![Heatmap Sample 4][heatmap4]
![Heatmap Sample 5][heatmap5]

### Here the resulting bounding boxes:
![Final Result0][final0]
![Final Result1][final1]
![Final Result2][final2]
![Final Result3][final3]
![Final Result4][final4]
![Final Result5][final5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline's robustness is not high because like bounding box size is not stable for same vehicle and there are still some false positives. To improve robustness I think I can use a kind of model of vehicle and its movement. Current approach is only find vehicles from scratch for each frame basically. I should be able to estimate vehicle's next frame's position after finding vehicle and its speed. And vehicle's size should not be changed without perspective, so I should be able to estimate size change as well. I think combination of current approach and tracking with vehicle's move and size model should be more robust.  
