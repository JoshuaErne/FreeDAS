import cv2
import numpy as np
import matplotlib.pyplot as plt
from Line import Line
import glob
import pickle

window_size = 5
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False
left_curve, right_curve = 0., 0.
left_lane_inds, right_lane_inds = None, None


class PolynomialLane:

    def __init__(self, frame, binary_warped):
        self.frame = frame
        self.binary_warped = binary_warped[0]
        self.m_inv = binary_warped[1]

    def polyLine(self, gray):
        img_size = (self.frame.shape[1], self.frame.shape[0])
        _, b_img = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        histogram = np.sum(b_img[b_img.shape[0] // 2:, :], axis=0)
        out_img = self.binary_warped
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9

        window_height = np.int(b_img.shape[0] / nwindows)

        nonzero = b_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 100

        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):

            win_y_low = b_img.shape[0] - (window + 1) * window_height
            win_y_high = b_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin - 10
            win_xright_high = rightx_current + margin - 10

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if not len(leftx) is 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            lc, res, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
            lfit = np.poly1d(lc)

        if not len(rightx) is 0:
            right_fit = np.polyfit(righty, rightx, 2)
            rc, res, _, _, _ = np.polyfit(righty, rightx, 2, full=True)
            rfit = np.poly1d(rc)

        out_img = cv2.resize(out_img, (640, 360))
        cv2.imshow('Sliding Window', out_img)

        if not len(leftx) is 0:
            left_fit = left_line.add_fit(left_fit)
        if not len(rightx) is 0:
            right_fit = right_line.add_fit(right_fit)


        if not len(leftx) is 0:
            y_eval = 719
            ym_per_pix = 5 / 720
            xm_per_pix = 0.3 / 1250
            try:
                left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
                    2 * left_fit[0])
                right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
                    2 * right_fit[0])
            except:
                return

        if not len(leftx) is 0:
            bottom_y = self.frame.shape[0] - 1
            bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
            bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]
            vehicle_offset = self.frame.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2
            lane_width = (bottom_x_right - bottom_x_left) / 2

            # xm_per_pix = 0.3/700
            # vehicle_offset *= xm_per_pix
            vehicle_offset = vehicle_offset * 100 / lane_width

        ploty = np.linspace(0, self.frame.shape[0] - 1, self.frame.shape[0])

        if not len(leftx) is 0:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            color_warp = np.zeros((720, 1280, 3), dtype='uint8')

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            # pts = np.array([[0, 435], [418, 275],[640, 411], [209, 275]])
            pts = np.array([[383, 194], [256, 192], [0,350], [640, 350]])
            # pts = np.array([[418, 275], [209, 275], [0, 435], [640, 411]])

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            newwarp = cv2.warpPerspective(color_warp, self.m_inv, (self.frame.shape[1], self.frame.shape[0]))

            result = cv2.addWeighted(self.frame, 1, newwarp, 0.3, 0)

            avg_curve = (left_curverad + right_curverad) / 2
            label_str_rad = 'Radius of curvature: %.1f cm' % avg_curve
            result = cv2.putText(result, label_str_rad, (30, 40), 0, 1, (255, 0, 0), 2, cv2.LINE_AA)

            label_str_offset = 'Vehicle offset from lane center: %.1f %%' % vehicle_offset
            result = cv2.putText(result, label_str_offset, (30, 70), 0, 1, (255, 0, 0), 2, cv2.LINE_AA)

            result = cv2.resize(result, (640, 360))

            return label_str_rad, label_str_offset
