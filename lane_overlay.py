import cv2 as cv
import numpy as np


def warp_back(original_image, warped_image, left_fitx, right_fitx, ploty, Minv):
    """

    Parameters:
    - original_image: Original
    - warped_image: Transformisana slika za detekciju
    - left_fitx: X koordinate za levu traku
    - right_fitx: X koordinate za desnu traku
    - ploty: Y koordinate za sve tacke traka
    - Minv: Inverzna matrica perspektivne transformacije

    Returns:
    - result: Kombinovana slika sa 'lane overlay' na originalnoj slici
    """
    # Prazna slika za crtanje traka
    lane_image = np.zeros_like(warped_image)

    # Pravljenje tačaka za poligone traka
    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left_points, right_points))

    # Crtanje traka
    cv.fillPoly(lane_image, np.int_([points]), 255)

    # Warp nazad u perspektivu originalne slike
    unwarped_lane = cv.warpPerspective(lane_image, Minv, (original_image.shape[1], original_image.shape[0]))

    # Kombinovanje sa originalnom slikom
    result = cv.addWeighted(original_image, 1, np.dstack((unwarped_lane,) * 3), 0.3, 0)

    return result
