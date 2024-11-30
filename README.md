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

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The CameraCalibration.py script is used to calibrate a camera by determining its intrinsic parameters and distortion coefficients.

First, we define the number of rows and columns in the chessboard pattern (9x6) and set the termination criteria for the corner sub-pixel refinement algorithm. We prepare a grid of object points corresponding to the expected positions of the chessboard corners, assuming that the chessboard lies on the ('x', 'y') plane at z=0. These object points are identical for all calibration images.

Next, we loop through all calibration images located in the camera_cal/ directory. Each image is converted to grayscale, and we attempt to find the corners of the chessboard pattern using the cv2.findChessboardCorners() function. If corners are successfully detected, their positions are refined using the corner sub-pixel refinement algorithm (cv2.cornerSubPix()). The detected corners are then drawn on the image, and a brief visual display allows us to verify the calibration process.

Once the object and image points are collected, we calibrate the camera using cv2.calibrateCamera(), obtaining the camera matrix (camera_matrix), distortion coefficients (dist_coeffs), and rotation and translation vectors (rvecs and tvecs). The calibration results are saved to a file named calib.npz.

To validate the calibration, we applied the distortion correction to a test image using cv2.undistort(). We also optimized the camera matrix for a valid field of view using cv2.getOptimalNewCameraMatrix(). The result is a corrected, undistorted image, which was optionally cropped to display the valid region of the frame.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

TODO: Add your text here!!!

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

