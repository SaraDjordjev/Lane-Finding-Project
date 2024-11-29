import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def detect_lane_pixels_and_fit(warped_image, n_windows=9, margin=100, minpix=50, visualize=False):
    """
    Detektuje piksele trake i fitovanje polinoma za granice trake.

    Args:
        warped_image (np.array): Transformisana binarna slika.
        n_windows (int): Broj klizecih prozora.
        margin (int): Sirina prozora za pretragu.
        minpix (int): Minimalan broj piksela za pomeranje prozora.
        visualize (bool): Da li prikazati rezultate.

    Returns:
        tuple: (left_fit, right_fit, ploty, left_fitx, right_fitx)
            - left_fit, right_fit: Koeficijenti fitovanih polinoma za levu i desnu traku.
            - ploty: Generisane y koordinate za crtanje linija.
            - left_fitx, right_fitx: X koordinate fitovanih linija.
    """
    # Kreiranje histograma na donjoj polovini slike
    histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :], axis=0)
    # Prikaz histograma
    plt.plot(histogram)
    plt.title("Histogram binarne slike")
    plt.xlabel("X koordinata")
    plt.ylabel("Broj belih piksela")
    plt.show()

    # Deljenje na levu i desnu polovinu
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if leftx_base == 0 and rightx_base == 0:
        print("Nijedna traka nije detektovana.")
    elif leftx_base == 0:
        print("Samo desna traka je detektovana.")
    elif rightx_base == 0:
        print("Samo leva traka je detektovana.")

    # Parametri prozora
    window_height = warped_image.shape[0] // n_windows
    nonzero = warped_image.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds = []
    right_lane_inds = []

    # Klizeci prozori
    for window in range(n_windows):
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Pronalazak piksela unutar prozora
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Pomeranje prozora
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Kombinovanje indeksa
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Koordinate piksela za levu i desnu traku
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # Fitovanje polinoma
    #left_fit = np.polyfit(lefty, leftx, 2)
    #right_fit = np.polyfit(righty, rightx, 2)

    # Generisanje y koordinata
    #ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
    #left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]


    ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])

    # Provera da li postoje dovoljno piksela za fitovanje
    if len(leftx) == 0 or len(lefty) == 0:
        print("Nema dovoljno piksela za levu traku.")
        left_fit = [0, 0, 0]
        left_fitx = np.zeros_like(ploty)
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    if len(rightx) == 0 or len(righty) == 0:
        print("Nema dovoljno piksela za desnu traku.")
        right_fit = [0, 0, 0]
        right_fitx = np.zeros_like(ploty)
    else:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #  if visualize:
  #      plt.imshow(warped_image, cmap='gray')
   #     plt.plot(left_fitx, ploty, color='yellow')
    #    plt.plot(right_fitx, ploty, color='yellow')
     #   plt.title("Detektovane vozne trake")
      #  plt.show()

    return left_fit, right_fit, ploty, left_fitx, right_fitx
