import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import imutils
from moviepy.editor import VideoFileClip


class BirdEyeView:

    def __init__(self, img, truth_value):
        self.img = img
        self.truth_value = truth_value

    def display(self):
        pass

    def unwarp(self):

        h, w = self.img.shape[:2]
        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        src = np.float32([(0, 410),
                          (225, 260),
                          (680, 410),
                          (410, 260)
                          ])

        dst = np.float32([(640, 0),
                          (0, 0),
                          (640, 480),
                          (0, 480)])

        M = cv2.getPerspectiveTransform(src, dst)

        # use cv2.warpPerspective() to warp your image to a top-down view
        warped1 = cv2.warpPerspective(self.img, M, (w, h), flags=cv2.INTER_LINEAR)
        warped = cv2.rotate(warped1, cv2.ROTATE_90_CLOCKWISE)

        if self.truth_value:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            f.subplots_adjust(hspace=.2, wspace=.05)
            ax1.imshow(self.img)
            x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
            y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
            ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
            ax1.set_ylim([h, 0])
            ax1.set_xlim([0, w])
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(warped)
            ax2.set_title('Unwarped Image', fontsize=30)
            plt.show()
        else:
            return warped, M




