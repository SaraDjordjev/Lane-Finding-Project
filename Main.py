import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import undistort_image, process_hsv, process_canny, perspective_transform
from lane_detection import detect_lane_pixels_and_fit
from lane_overlay import warp_back

def process(image):
    """Glavna funkcija za obradu slike."""
    # Ispravljanje distorzije slike
    undistorted_image = undistort_image(image)

    # Procesiranje HSV maske
    filtered_image, combined_mask = process_hsv(undistorted_image)

    # Detekcija ivica pomocu maske
    edges = process_canny(combined_mask, thr1=30, thr2=200)

    # Dodavanje tacaka za perspektivnu transformaciju
    src = np.float32([
        [600, 460],
        [720, 460],
        [1140, 720],
        [220, 720]
    ])
    dst = np.float32([
        [300, 0],
        [1000, 0],
        [1000, 720],
        [300, 720]
    ])

    warped_image, points_image, M, Minv = perspective_transform(edges, src, dst)
    # Detekcija piksela trake i fitovanje
    left_fit, right_fit, ploty, left_fitx, right_fitx = detect_lane_pixels_and_fit(warped_image, visualize=True)

    # Warp lane back onto the original image
    lane_overlay = warp_back(undistorted_image, warped_image, left_fitx, right_fitx, ploty, Minv)

    return undistorted_image, filtered_image, edges, warped_image, points_image, left_fitx, right_fitx, ploty, lane_overlay

def main():

    input_path = 'test_videos/project_video03.mp4'
    is_video = input_path.endswith(('.mp4', '.avi', '.mov'))  # Provera da li je ulaz video

    if is_video:

        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            print("Greska: Video nije pronadjen.")
            return

        # Postavke za zapisivanje izlaznog videa
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesiranje okvira
            results = process(frame)
            _, _, _, warped_image, _, _, _, _, lane_overlay = results
            plt.imshow(warped_image, cmap='gray')
            plt.title("Ulazna binarna slika")
            plt.show()

            # Prikaz rezultata
            cv.imshow('Lane Detection', lane_overlay)
            out.write(lane_overlay)

            if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' za prekid
                break

        cap.release()
        out.release()
        cv.destroyAllWindows()
        print("Procesiranje videa zavrseno.")
    else:

        image = cv.imread(input_path)
        if image is None:
            print("Greska: Slika nije pronadjena.")
            return

        # Procesiranje slike
        results = process(image)
        undistorted, filtered, edges, warped, points_image, left_fitx, right_fitx, ploty, lane_overlay = results


        # Prikaz rezultata
        cv.imshow('Original', image)
        cv.imshow('Filtered Image', filtered)
        cv.imshow('Edges', edges)
        cv.imshow('Warped Image', warped)
        cv.imshow('Points Image', points_image)


        cv.imwrite('Binary.jpg', edges)
        cv.imwrite('warped_image.jpg', warped)

        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imshow('Lane Detection', lane_overlay)
        cv.imwrite('output_image.jpg', lane_overlay)
        print("Procesiranje slike zavrseno.")
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
