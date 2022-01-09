import numpy as np
import cv2 as cv

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv.VideoCapture('Data/outpy.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# Tuned best value
Gamma = 0.7
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, Gamma) * 255.0, 0, 255)
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, img = cap.read()
  if ret == True:

    # Gamma correction
    road = cv.LUT(img, lookUpTable)
    gray = cv.cvtColor(road,cv.COLOR_BGR2GRAY)
    
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(road,markers)
    
    locs = np.logical_or(markers == 1,markers == 3)
    locs = np.logical_or(locs,markers == -1)
    locs = np.logical_or(locs,markers == 0)
    road[locs] = [255,0,0]

    height, width, ch = img.shape
    new_width, new_height = width + width/20, height + height/8

    # Crate a new canvas with new width and height.
    canvas_orig = np.ones((int(new_height), int(new_width), ch), dtype=np.uint8) * 125

    # New replace the center of canvas with original image
    padding_top, padding_left = 60, 10
    if padding_top + height < new_height and padding_left + width < new_width:
        canvas_orig[padding_top:padding_top + height, padding_left:padding_left + width] = img
    else:
        print("The Given padding exceeds the limits.")

    # Crate a new canvas with new width and height.
    canvas_road = np.ones((int(new_height), int(new_width), ch), dtype=np.uint8) * 125

    # New replace the center of canvas with original image
    padding_top, padding_left = 60, 10
    if padding_top + height < new_height and padding_left + width < new_width:
        canvas_road[padding_top:padding_top + height, padding_left:padding_left + width] = road
    else:
        print("The Given padding exceeds the limits.")

    text1 = "Original"
    text2 = "Road"
    img1 = cv.putText(canvas_orig.copy(), text1, (int(0.25*width), 30), cv.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])
    img2 = cv.putText(canvas_road.copy(), text2, (int(0.25*width), 30), cv.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])

    final = cv.hconcat((img1, img2))
    cv.imshow("result", final)
    cv.waitKey(1)
