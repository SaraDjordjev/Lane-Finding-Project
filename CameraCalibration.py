import cv2
import numpy as np
import glob

rows = 6
cols = 9
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

objpoints = []
imgpoints = []

folder_path = "camera_cal"
images = glob.glob(f"{folder_path}/*.jpg")


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objectPoints)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(20)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('camera_cal/calib.npz', mtx=camera_matrix, dist=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
#print("Matrica kamere:\n", camera_matrix)
#print("Koeficijenti distorzije:\n", dist_coeffs)


######################################################################
# Primer korekcije slike
img = cv2.imread('camera_cal/calibration1.jpg')
h, w = img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

x, y, w, h = roi
cropped_img = undistorted_img[y:y+h, x:x+w]

# Prikaz originalne i ispravljene slike
cv2.imshow('Originalna', img)
cv2.imshow('Ispravljena', undistorted_img)
cv2.imshow('Ispravljena i izrezana slika', cropped_img)
cv2.imwrite('examples/calibration.jpg', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
