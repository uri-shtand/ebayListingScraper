# This is loosely based on code by Adrian from PyImageSearch
# import the necessary packages
import time

import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream

import dlib

displayWidth = 900
prototxt = "./model/deploy.prototxt"
model = "./model/weights.caffemodel"
maxConfidence = 0.5
predictorPath = "./model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictorPath)


def applyFacialPattern(head, degrees):
    (h, w) = head.shape[:2]
    rect = dlib.rectangle(0, 0, w, h)
    shape = predictor(head, rect)
    shape = face_utils.shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(head, (x, y), 1, (0, 0, 255), -1)
    return head


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
degrees = 0
direction = 1
filterType = 0


def verifyFaces(detections, frame):
    (h, w) = frame.shape[:2]
    result = []
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
        result.append(box)
    return result


def findFacialPatterns(boxedFaces, frame):
    patterns = []
    for box in boxedFaces:
        (startX, startY, endX, endY) = box.astype("int")
        rect = dlib.rectangle(startX, startY, endX, endY)
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        patterns.append((shape, box))
    return patterns


def markValidDetections(facialPatterns, frame):
    for (shape, box) in facialPatterns:
        (startX, startY, endX, endY) = box.astype("int")
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


while True:
    degrees = degrees + direction
    if degrees > 9 or degrees == 0:
        direction = 0 - direction
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = videoStream.read()
    frame = imutils.resize(frame, width=displayWidth)

    detections = findFaces(frame)
    boxedFaces = verifyFaces(detections, frame)
    facialPatterns = findFacialPatterns(boxedFaces, frame)
    markValidDetections(facialPatterns, frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord('f'):
        filterType = filterType + 1

# do a bit of cleanup
cv2.destroyAllWindows()
videoStream.stop()
