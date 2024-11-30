import cv2 as cv
import numpy as np

calibration_data = np.load('camera_cal/calib.npz')
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

def undistort_image(image):
    """Ispravljanje distorzije slike koristeci kalibracione parametre."""
    undistorted = cv.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
    return undistorted

def process_hsv(image):
    """Primena HSV maske za detekciju belih i zutih delova slike."""
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Definisanje granica za boje
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([255, 80, 255])

    yellow_lower = np.array([15, 90, 90])
    yellow_upper = np.array([35, 255, 255])

    # Pravljenje maski
    white_mask = cv.inRange(hsv_image, white_lower, white_upper)
    yellow_mask = cv.inRange(hsv_image, yellow_lower, yellow_upper)

    # Kombinovanje maski pomocu OR operacije
    combined_mask = cv.bitwise_or(white_mask, yellow_mask)

    # Aplikacija maske na originalnu sliku
    masked_image = cv.bitwise_and(image, image, mask=combined_mask)

    return masked_image, combined_mask

def process_canny(mask, thr1, thr2):
    """Primena Canny algoritma za detekciju ivica."""
    blurred = cv.GaussianBlur(mask, (5, 5), 0)
    edges = cv.Canny(blurred, thr1, thr2)
    return edges


def perspective_transform(binary_image, src_points, dst_points):
    """
    Primena perspektivne transformacije za pticju perspektivu.

    Args:
        binary_image (np.array): Ulazna binarna slika.
        src_points (np.array): Izvorne tacke za transformaciju.
        dst_points (np.array): Ciljne tacke za transformaciju.

    Returns:
        np.array: Slika transformisana u pticju perspektivu.
        np.array: Slika sa nacrtanim tackama.
        np.array: Matrica perspektivne transformacije.
        np.array: Inverzna matrica perspektivne transformacije.
    """
    # Kreiranje matrica transformacije
    M = cv.getPerspectiveTransform(src_points, dst_points)
    Minv = cv.getPerspectiveTransform(dst_points, src_points)  # Inverzna matrica

    warped_image = cv.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))

    image_with_points = cv.polylines(binary_image.copy(), [np.int32(src_points)], isClosed=True, color=255, thickness=3)

    return warped_image, image_with_points, M, Minv
