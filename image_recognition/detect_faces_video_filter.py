# This is based on code by Adrian from PyImageSearch
# import the necessary packages
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

maxWidth = 900
prototxt = "./model/deploy.prototxt"
model = "./model/weights.caffemodel"
maxConfidence = 0.5


def markValidDetections(detections, frame, degrees, filterType):
    (h, w) = frame.shape[:2]
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < maxConfidence:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        filter = switchFilter(filterType)
        applyFilterOnHead(endX, endY, frame, startX, startY, degrees, filter)


def applyFilterOnHead(endX, endY, frame, startX, startY, degrees, filter):
    head = frame[startY:endY, startX:endX]
    w = endX - startX
    h = endY - startY
    filteredHead = filter(head, degrees)
    frame[startY:startY + h, startX:startX + w] = filteredHead


def applyBlurFilterOnFace(head, degrees):
    return cv2.GaussianBlur(head, (degrees, degrees), 0)


def applyGrayscaleFilterOnFace(head, degrees):
    filteredHead = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(filteredHead, cv2.COLOR_GRAY2RGB)


def applyEdgeFilter(head, degrees):
    filteredHead = cv2.Canny(head, 30, 150)
    return cv2.cvtColor(filteredHead, cv2.COLOR_GRAY2RGB)


def switchFilter(argument):
    switcher = {
        0: applyEdgeFilter,
        1: applyGrayscaleFilterOnFace,
        2: applyBlurFilterOnFace
    }
    return switcher.get(argument, lambda: "Invalid filter")


def findFaces(frame):
    # grab the frame dimensions and convert it to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    return detections


# MAIN starts here
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream... PRESS q to exit")
videoStream = VideoStream(src=0).start()
# Allow the sensor time to warm
time.sleep(2.0)

# loop over the frames from the video stream
degrees = 5
direction = 2
filterType = 0
while True:
    degrees = degrees + direction
    if degrees > 21 or degrees == 5:
        direction = 0 - direction
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = videoStream.read()
    frame = imutils.resize(frame, width=maxWidth)

    detections = findFaces(frame)
    markValidDetections(detections, frame, degrees, filterType)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord('f'):
        filterType = (filterType + 1) % 3

# do a bit of cleanup
cv2.destroyAllWindows()
videoStream.stop()
