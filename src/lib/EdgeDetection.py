import cv2
import numpy as np
import matplotlib.pyplot as plt
import PerspectiveTransform
from PerspectiveTransform import BirdEyeView
from lane_detection import PolynomialLane


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = np.array([0, 0, 0, 0])
    right_line = np.array([0, 0, 0, 0])
    if left_fit:
        left_fit_average = np.average(left_fit, axis = 0)
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis = 0)
        right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def canny(image_cn):
    gray = cv2.cvtColor(image_cn, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            try:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            except:
                continue
    return line_image


def region_of_interest(image_roi, polygons = np.array([
[(100, 352), (560,357),(375, 250),(275, 241)]
])):
    height = image_roi.shape[0]
    mask = np.zeros_like(image_roi)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image_roi, mask)
    return masked_image
    


class LaneDetection:
    def __init__(self):
        pass

    def video_capture(self):
        self.cap = cv2.VideoCapture("Data/train.mp4")

    def canny_edge_detection(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_image = cv2.Canny(blur, 50, 150)
        persp_obj = BirdEyeView(frame, False)
        perspective_image = persp_obj.unwarp()

        gray_perspective = cv2.cvtColor(perspective_image[0], cv2.COLOR_RGB2GRAY)
        blur_perspective = cv2.GaussianBlur(gray_perspective, (5, 5), 0)
        canny_perspective = \
            cv2.Canny(blur_perspective, 50, 150)

        cropped_image = region_of_interest(canny_image)

        # Calling lane_detection for sliding window
        poly_lane_detect = PolynomialLane(frame, perspective_image)
        try:
            radius, offset = poly_lane_detect.polyLine(gray_perspective)
        except:
            radius, offset = None, None

        # cropped_perspective = region_of_interest(canny_perspective, np.array([
        #     [(100, 352), (560, 357), (375, 250), (275, 241)]
        # ]))

        cropped_perspective = region_of_interest(canny_perspective, np.array([
            [(20, 550), (100, 50), (450, 550), (400, 50)]
        ]))

        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 20, np.array([]), minLineLength=10, maxLineGap=1000)

        perspective_lines = cv2.HoughLinesP(cropped_perspective, 2, np.pi / 180, 20, np.array([]), minLineLength=10,
                                            maxLineGap=1000)

        if lines is not None:
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)

            if perspective_lines is not None:
                averaged_lines = average_slope_intercept(perspective_image[0], perspective_lines)
                perspective_line_image = display_lines(perspective_image[0], averaged_lines)
            else:
                perspective_line_image = None

            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
            
            if perspective_lines is not None:
                combo_image = cv2.addWeighted(perspective_image[0], 0.8, perspective_line_image, 1, 0)
                cv2.imshow("Sliding Window", combo_image)

        else:
            line_image = None
            perspective_line_image = None


        return line_image, radius, offset
