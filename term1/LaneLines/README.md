# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[img-step1]: ./images/solidYellowCurve-step1.jpg "Step1"
[img-step2]: ./images/solidYellowCurve-step2.jpg "Step2"
[img-step3]: ./images/solidYellowCurve-step3.jpg "Step3"
[img-step4]: ./images/solidYellowCurve-step4.jpg "Step4"
[img-step5]: ./images/solidYellowCurve-step5.jpg "Step5"
[img-result]: ./images/solidYellowCurve-result.jpg "Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps as follows.
1. Converted the images to grayscale.
![step1][img-step1]
2. Applied gaussian blur to the grayscale images to reduce noises.
![step2][img-step2]
3. Detected edges by Canny method from smoothed grayscale images.
![step3][img-step3]
4. Applied shape mask with trapezoid shape to edge images.
![step4][img-step4]
5. Transformed by Hough method from masked edge images to lines.
![step5][img-step5]

Then output image would be like following after joining pipelines result and original image.

![result][img-result]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by slope, angle filtering and K-means clustering.

If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when road is sharp curve.  

Another shortcoming could be turning.

Because my pipelines uses angle filtering, the threshold should be changed for curve and turning if the filtering will be used in those situation as well.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement road type detection and change parameters for each road.

My pipelines work only for straight and gentle curve because parameters are tuned only for those type of road.

To fit my pipelines to another type of road, road type detection and use tuned pipelines for each road.

Another potential improvement could be to use high dimension fitting method.

I only find lines in my pipelines by Hough Transform. But, in real road, lanes should be curve in many cases.

And curve fitting method would be able to fit even if road is sharp curve or car is turning.
