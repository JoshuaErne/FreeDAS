from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core.numeric import indices

import _init_paths

from opts import opts
from detectors.detector_factory import detector_factory

# import speed estimation model
from lib.SpeedEstimator import Realtimespeed, get_annotated_frame, neural_factory

# import lane detection module
from lib.EdgeDetection import LaneDetection

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

SIFT_params = dict( maxCorners = 100,
                      qualityLevel = 0.1,
                      minDistance = 7,
                      blockSize = 1)

# Parameters for lucas kanade optical flow
KLT_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for tracking
color = np.random.randint(0, 255, (100, 3))

# Video framerate
FPS = 20

# Seconds per Detection
SPD = 1

def im_pretty_show(prediction_annotation, img1):
  road = prediction_annotation
  height, width, ch = img1.shape
  new_width, new_height = width + width/20, height + height/8

  # Crate a new canvas with new width and height.
  canvas_orig = np.ones((int(new_height), int(new_width), ch), dtype=np.uint8) * 125

  # New replace the center of canvas with original image
  padding_top, padding_left = 60, 10
  if padding_top + height <= new_height and padding_left + width <= new_width:
      canvas_orig[padding_top:padding_top + height, padding_left:padding_left + width] = img1
  else:
      print("The Given padding exceeds the limits.")

  # Crate a new canvas with new width and height.
  canvas_road = np.ones((int(new_height), int(new_width), ch), dtype=np.uint8) * 125

  # New replace the center of canvas with original image
  padding_top, padding_left = 60, 10
  if padding_top + height <= new_height and padding_left + width <= new_width:
      canvas_road[padding_top:padding_top + height, padding_left:padding_left + width] = road
  else:
      print("The Given padding exceeds the limits.")

  text1 = "Original"
  text2 = "Prediction"
  texted_image1 = cv2.putText(canvas_orig.copy(), text1, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])
  texted_image2 = cv2.putText(canvas_road.copy(), text2, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])

  final = cv2.hconcat((texted_image1, texted_image2))
  cv2.imshow("result", final)
  
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  DetectorClass = detector_factory[opt.task]
  objectDetector = DetectorClass(opt)

  # # Initialize Speed Estimation Model
  # speed_model = neural_factory()
  # speed_model.load_weights('models/speed_model.h5')

  # # Speed Estimation Parameters
  # frame_idx = 1
  # y_true = [0,0]

  # Initialize Lane Detector Class
  laneDetector = LaneDetection()
  
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    objectDetector.pause = False
    # ret, img_old = cam.read()
    frame_idx = 0
    while True:
      if frame_idx % (SPD * FPS) == 0:
        ret, img_new = cam.read()
        if not ret:
            print('No frames grabbed!')
            break

        # img_old_gray   = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        detection_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        
        # prediction_annotation, y_true = get_annotated_frame(detection_gray, img_old_gray, frame_idx, speed_model, y_true)
      
        # im_pretty_show(prediction_annotation, img_new)

        line_image, radius, offset = laneDetector.canny_edge_detection(img_new)
        try:
          img_new = cv2.addWeighted(img_new, 0.8, line_image, 1, 0)
          img_new = cv2.putText(img_new, radius, (30, 480-40), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
          img_new = cv2.putText(img_new, offset, (30, 480-70), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
        except:
          pass
        cv2.imshow('input', img_new)
        
        result, debugger = objectDetector.run(img_new)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, result[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit

      elif (frame_idx - 1) % (SPD * FPS) == 0:
        ret, img_new = cam.read()
        if not ret:
            print('No frames grabbed!')
            break
          
        cv2.imshow('input', img_new)

        mask = np.zeros_like(img_new)
        mask, centers, indices = objectDetector.create_detections_mask(debugger, mask, result['results'])
        
        current_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
  
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(detection_gray, current_gray, centers, None, **KLT_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = centers[st==1]

        # Create a mask image for drawing purposes
        tracker = np.zeros_like(img_new)
        tracked = img_new.copy()
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            tracker = cv2.line(tracker, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            tracked = cv2.circle(tracked, (int(a), int(b)), 5, color[i].tolist(), -1)

        tracked = cv2.add(tracked, tracker)

        line_image, radius, offset = laneDetector.canny_edge_detection(img_new)
        try:
          # tracked = cv2.addWeighted(tracked, 0.8, line_image, 1, 0)
          tracked = cv2.putText(tracked, radius, (0, 480-40), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
          tracked = cv2.putText(tracked, offset, (0, 480-70), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
        except:
          pass

        objectDetector.custom_show_results(debugger, tracked, result['results'])

        # cv2.imshow('ctdet', tracked)
        if cv2.waitKey(1) == 27:
            return  # esc to quit

        # Now update the previous frame and previous points
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

      else:
        _, img_new = cam.read()
        cv2.imshow('input', img_new)
        
        current_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **KLT_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        tracked = img_new.copy()
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            tracker = cv2.line(tracker, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            tracked = cv2.circle(tracked, (int(a), int(b)), 5, color[i].tolist(), -1)

        tracked = cv2.add(tracked, tracker)

        line_image, radius, offset = laneDetector.canny_edge_detection(img_new)
        try:
          tracked = cv2.addWeighted(tracked, 0.8, line_image, 1, 0)
          tracked = cv2.putText(tracked, radius, (30, 480-40), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
          tracked = cv2.putText(tracked, offset, (30, 480-70), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
        except:
          pass

        # This is prone to corrupting boxes, it is not perfect
        # objectDetector.update_boxes(good_new, result['results'], indices)
        # objectDetector.custom_show_results(debugger, tracked, result['results'])

        # Show tracked centers only
        cv2.imshow('ctdet', tracked)

        if cv2.waitKey(1) == 27:
            return  # esc to quit

        # Now update the previous frame and previous points
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

      frame_idx += 1
      img_old = img_new
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      result = objectDetector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, result[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
