**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
--------------------------------------------------------------------------------------------------------------------------

1. **CameraCalibration.py**: Handles camera calibration and image undistortion.
2. **preprocessing.py**: Performs preprocessing steps such as color filtering, edge detection, and perspective transformation.
3. **lane_detection.py**: Detects lane pixels and fits a polynomial to represent the lanes.
4. **lane_overlay.py**: Overlays the detected lanes back onto the original image.

The `main.py` script integrates all these components to process videos or images, detecting lanes and 
outputting the result as a video or image. The program supports both real-time lane visualization and 
video processing, making it suitable for testing under various conditions.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The CameraCalibration.py script is used to calibrate a camera by determining its intrinsic parameters and distortion coefficients.

First, we define the number of rows and columns in the chessboard pattern (9x6) and set the termination criteria for the corner sub-pixel refinement algorithm. We prepare a grid of object points corresponding to the expected positions of the chessboard corners, assuming that the chessboard lies on the ('x', 'y') plane at z=0. These object points are identical for all calibration images.

Next, we loop through all calibration images located in the camera_cal/ directory. Each image is converted to grayscale, and we attempt to find the corners of the chessboard pattern using the `cv2.findChessboardCorners()` function. If corners are successfully detected, their positions are refined using the corner sub-pixel refinement algorithm (`cv2.cornerSubPix()`). The detected corners are then drawn on the image, and a brief visual display allows us to verify the calibration process.

Once the object and image points are collected, we calibrate the camera using `cv2.calibrateCamera()`, obtaining the camera matrix (`camera_matrix`), distortion coefficients (`dist_coeffs`), and rotation and translation vectors (`rvecs` and `tvecs`). The calibration results are saved to a file named `calib.npz`.

To validate the calibration, we applied the distortion correction to a test image using `cv2.undistort()`. We also optimized the camera matrix for a valid field of view using `cv2.getOptimalNewCameraMatrix()`. The result is a corrected, undistorted image, which was optionally cropped to display the valid region of the frame.

![Original Image](camera_cal/calibration1.jpg)
![Corrected Image](examples/calibration.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The `preprocessing.py` script includes multiple techniques for creating a thresholded binary image. These methods involve color transforms and edge detection, providing a robust way to isolate regions of interest in an image.

**Color Transform**
In the `process_hsv()` function, we use the HSV color space to detect specific colors (yellow and white) in the image. This involves:

1. Converting the input image to the HSV color space using `cv.cvtColor(image, cv.COLOR_BGR2HSV)`.
2. Defining lower and upper bounds for the yellow and white colors.
    White: [0, 0, 180] to [255, 80, 255]
    Yellow: [15, 90, 90] to [35, 255, 255]
3. Creating binary masks using `cv.inRange()`, which thresholds the image to include only the pixels within the defined color ranges.
4. Combining the masks with a bitwise OR operation: `cv.bitwise_or(white_mask, yellow_mask)`.
5. Applying the combined mask to the original image using `cv.bitwise_and()`.
   
The output of this step includes both the masked image and the binary mask used for thresholding.
```python
 masked_image, combined_mask = process_hsv(image)
```
**Edge Detection**
In the `process_canny()` function, we use the Canny edge detection algorithm:

1. The binary mask from `process_hsv()` is blurred using a Gaussian filter (`cv.GaussianBlur()`).
2. Canny edge detection is applied using `cv.Canny()` with adjustable thresholds (thr1 and thr2). This isolates edges based on intensity gradients in the image.
```python
edges = process_canny(combined_mask, thr1, thr2)
```
**Result**
The final binary image is a combination of the thresholding and edge detection steps, where regions corresponding to the desired colors (yellow and white) and strong edges are highlighted.

Below is an example of the resulting binary image:

![Binary Image](examples/Binary.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the `preprocessing.py` script, the `perspective_transform()` function is used to apply a perspective transform. This technique is crucial for converting an image into a bird's-eye view, which is particularly useful in applications like lane detection.

**Steps of the Perspective Transform**
1. Source Points:
The input `src_points` represent the coordinates of the region in the original image that you want to transform.

2. Destination Points:
The `dst_points` represent the coordinates of the desired rectangular region in the output image.

3. Transformation Matrices:
Using `cv.getPerspectiveTransform()`, the function calculates the transformation matrix `M` and its inverse `Minv`. These matrices map the source points to the destination points (and vice versa for the inverse matrix).
```python
M = cv.getPerspectiveTransform(src_points, dst_points)
Minv = cv.getPerspectiveTransform(dst_points, src_points)
```

5. Applying the Transform:
The perspective transformation is applied to the binary input image using `cv.warpPerspective()`, which remaps the pixels according to the matrix `M`.
```python
warped_image = cv.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))
```
6. Visualization of Points:
To confirm the transformation region, the function draws the source points on the original binary image using `cv.polylines()`.
```python
image_with_points = cv.polylines(binary_image.copy(), [np.int32(src_points)], isClosed=True, color=255, thickness=3)
```
Below is an example of the perspective transformation:
![Wraped Image](examples/warped_image.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This project detects lane-line pixels in a warped binary image and fits their positions with a polynomial to define the boundaries of the lanes. The implemented algorithm utilizes a histogram-based sliding window approach to identify lane pixels and then applies polynomial fitting to create a smooth curve representing the lanes. (`lane_detection.py`)
**How Lane-Line Pixels Are Identified and Fit with a Polynomial**

**1. Create a Histogram of the Warped Binary Image**

The histogram is calculated using the bottom half of the warped binary image, summing pixel values along each column to identify the likely starting points of lane lines.
```python
histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :], axis=0)
plt.plot(histogram)
```
The peak of the histogram indicates the base positions of the left and right lane lines.

![Histogram Image](examples/Histogram.jpg)

**2. Divide Image and Locate Lane Base Points**

The midpoint of the histogram splits the image into left and right halves, and the peaks in each half determine the base positions of the lane lines.
```python
midpoint = histogram.shape[0] // 2
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
**3. Sliding Window Search**

A sliding window approach is used to iteratively search for lane pixels starting from the base points. For each window:
    - Pixels within the window are added to the lane line indices.
    - The window is moved vertically and adjusted horizontally based on the average x-coordinate of detected pixels.
```python
for window in range(n_windows):
    win_y_low = warped_image.shape[0] - (window + 1) * window_height
    win_y_high = warped_image.shape[0] - window * window_height
    good_left_inds = ...
    good_right_inds = ...
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
```
**4. Extract Pixel Coordinates**

The indices of detected pixels are used to extract their x and y coordinates.
```python
leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
```
**5. Fit a Polynomial**

A second-degree polynomial is fit to the detected pixels for both the left and right lanes using `numpy.polyfit`.
```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
**6. Generate Smooth Lane Curves**

Y-coordinates are generated, and the corresponding x-coordinates are calculated using the polynomial coefficients.
```python
ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
```
**7. Visualization**

The detected lanes and their polynomial fits can be visualized using Matplotlib.
```python
if visualize:
    plt.imshow(warped_image, cmap='gray')
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.show()
```
![Lane Detection](examples/LaneDetection.jpg)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![Output Image](examples/output_image.jpg)

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

![Output Video](examples/output_video.avi)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**1. Challenges Faced During Implementation**

The main issues encountered during the development of the lane detection pipeline were:
- Faint or Missing Lane Lines: Some test videos (`challenge02` and `challenge03`) had lanes that were hard to detect.
- Lighting Changes: Sudden changes in lighting, such as shadows or reflections, affected the accuracy of lane detection.
- Complex Roads: The pipeline struggled with sharp curves, merging lanes, or unusual road structures.

**2. Pipeline Performance and Failures**

The pipeline worked well on simpler videos (`project_video01`, `project_video02`, and `project_video03`) and moderately challenging ones (`challenge01`). However, it failed in more difficult scenarios (`challenge02` and `challenge03`) where conditions were too complex.

**3. Improvements for Robustness**

To make the pipeline more reliable, these improvements could help:
- Use adaptive thresholding to handle lighting changes better.
- Add tracking from previous frames to predict lane positions when detection is uncertain.
- Explore machine learning techniques to improve lane detection in challenging conditions.
  
**4. Conclusion**
The pipeline is effective for simple and moderately challenging conditions but needs further improvements to handle complex scenarios.
