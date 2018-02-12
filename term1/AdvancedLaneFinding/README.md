**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist]: ./output_images/undistortion.png "Undistorted Image"
[sample]: ./output_images/undistortion-lane.png "Sample Image"
[binary]: ./output_images/binarizing.png "Binary Example"
[unwarp]: ./output_images/unwarp.png "Warp Example"
[centroids]: ./output_images/window_centroids.png "Finding Lane Line Pixels Example"
[fitting]: ./output_images/lane_fitting.png "Fitting Lane Line Example"
[finalex]: ./output_images/lane_detection_result.png "Lane Detection Example"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

All my codes are located in advanced_lane_lines.ipynb and it's IPython notebook.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This part codes are in "Camera Calibration" and "Undistortion" section in the notebook.

At first I thought about object points which are ideal coordinates of the chessborad corners in the world. I set depth(z) = 3 same as lecture video because any positive small value is accepted.

About x and y, I just set x = 0 for left most corner and y = 0 for top most corner. And then adding 1 for next corners.

After that, I found chessborad corners from image using `cv2.findChessboardCorners` function and get mapping information between corner positions and real world positions.

I could get undistorted image using the mapping data and `cv2.undistort` function after calibration with `cv2.calibrateCamera` function. A sample undistortion result is following:

![Undistorted Image][undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Sample Image][sample]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary images. This codes are in "Binarizing by Color and Gradients" section.

Lane color is yellow or white, so I used "b" of LAB color space to find yellow and brightness from HLS color space to detect white. In addition to it, I also used gradient to improve accuracy because color is affected by light.

Here is the sample image:

![Binary Example][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform's source coordinates are generated by `calculate_roi_vertices` in "Other Preprocessing" section. This function generate a trapezoid vertices from triangle and offset values. How to generate is following:

```python
ysize, xsize = img.shape[0], img.shape[1]
left_top     = [xsize/2 - top_offset[0], ysize/2 + top_offset[1]]
right_top    = [xsize/2 + top_offset[0], ysize/2 + top_offset[1]]
left_bottom  = [        bottom_offset[0], ysize - bottom_offset[1]]
right_bottom = [xsize - bottom_offset[0], ysize - bottom_offset[1]]
```

About destination, basically image's the four corners. But I used a offset value for x position to reduce distortion. This code is in "Perspective Transform" section.

```python
src = np.float32([left_top, right_top, left_bottom, right_bottom])
dst = np.float32([[x_offset, 0], [xsize - x_offset, 0], [x_offset, ysize], [xsize - x_offset, ysize]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 540, 460      | 200,  0       |
| 740, 460      | 1080, 0       |
| 50, 670       | 200,  720     |
| 1230, 670     | 1080, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective Transform Example][unwarp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

At "Finding Window Centroids" section, I found lane line pixels like following:

![Finding Lane Line Pixels][centroids]

After that I fit lane lines with a 2nd order polynomial like following using window centroids:

![Fitting Lane Line][fitting]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated curvature using `calculate_curvature` function in "Find Fitting Line" section. The function estimate a radii from a fitting lines.

About position, I calculated it in `detect` function as a class method in "Draw Lane Area & Pipeline" section. The function calculates x axis difference between center of lanes and center of image (this should be the center of a car).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`detect` function is the my pipeline to find lane lines from image. Here is an example of the pipline:

![Lane Detection Example][finalex]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a my pipline result on a video.
[Pipeline Video Result](./output_images/project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach which I took needs many parameter tuning which effects accuracy strongly and doesn't work in night because of color and gradient thresholding. The approach's accuracy depends on binary image quality,

Accuracy with color and gradients thresholding is not good when it's night or there is dirt or something even if I use hue or saturation of HLS.

To improve robustness, it's better to use a kind of wide area shape based method for generating binary image. Gradient is kind of shape based method but it only care very small area so that it find dirt as lane line pixel. 