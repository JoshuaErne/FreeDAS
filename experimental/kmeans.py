import numpy as np
import cv2 as cv

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv.VideoCapture('Data/outpy.avi')
# cap = cv.VideoCapture('Data/train.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

tuned_centers = np.array([[ 65,  54,  45], 
                          [144, 126, 115], 
                          [ 84,  72,  61]])
                          
if(cap.isOpened()):
  # Capture frame-by-frame
  ret, img = cap.read()
else:
    exit()

# Tuned best value
Gamma = 0.4
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, Gamma) * 255.0, 0, 255)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Read until video is completed
center = None
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, img = cap.read()
  if ret == True:
    # # Histogram Equalization
    # img[:,:, 0] = clahe.apply(img[:,:, 0])
    # img[:,:, 1] = clahe.apply(img[:,:, 1])
    # img[:,:, 2] = clahe.apply(img[:,:, 2])

    # Gamma / Contrast Correction
    img = cv.LUT(img, lookUpTable)

    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Tuneable => K in kmeans
    K = 10
    ret,label,center=cv.kmeans(Z,K,center,criteria,1,cv.KMEANS_USE_INITIAL_LABELS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    
    # Sort by brightness => take brightest K-1 and keep darkest (cars)
    idx = np.argsort(np.linalg.norm(center,axis=1))
    center[idx[1:]] = np.array([255,255,255])
    center[idx[0]] = np.array([0,0,0])

    res = center[label.flatten()]
    road = res.reshape((img.shape))
    
    height, width, ch = img.shape
    new_width, new_height = width + width/20, height + height/8
    
    # Crate a new canvas with new width and height.
    canvas_orig = np.ones((int(new_height), int(new_width), ch), dtype=np.uint8) * 125

    # New replace the center of canvas with original image
    padding_top, padding_left = 60, 10
    if padding_top + height <= new_height and padding_left + width <= new_width:
        canvas_orig[padding_top:padding_top + height, padding_left:padding_left + width] = img
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
    text2 = "Road"
    img1 = cv.putText(canvas_orig.copy(), text1, (int(0.25*width), 30), cv.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])
    img2 = cv.putText(canvas_road.copy(), text2, (int(0.25*width), 30), cv.FONT_HERSHEY_COMPLEX, 1, [255, 0, 0])

    final = cv.hconcat((img1, img2))
    cv.imshow("result", final)
    cv.waitKey(1)