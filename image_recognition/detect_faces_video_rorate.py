# This is based on code by Adrian from PyImageSearch
# import the necessary packages
import argparse
import time
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream

maxWidth = 600
prototxt = "./model/deploy.prototxt"
model = "./model/weights.caffemodel"
maxConfidence = 0.5

def markValidDetections(detections, frame,degrees):
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
        #markDetection(confidence, endX, endY, frame, startX, startY)
        rotateFace(endX,endY,frame,startX,startY,degrees)

def rotateFace(endX, endY, frame, startX, startY,degrees):
    head = frame[startY:endY, startX:endX]
    w = endX - startX
    h = endY - startY
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -degrees, 1.0)
    rotatedHead = cv2.warpAffine(head, M, (w, h))
    (nh, nw) = rotatedHead.shape[:2]
    #print("startX: {} startY:{}, nh: {}, nw: {}, h: {}, w: {}".format(startX,startY,nh,nw,h,w))
    frame[startY:startY+nh, startX:startX+nw] = rotatedHead

def markDetection(confidence, endX, endY, frame, startX, startY):
    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
    cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


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
while True:
    degrees = degrees + 10 % 360
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = videoStream.read()
    frame = imutils.resize(frame, width=maxWidth)

    detections = findFaces(frame)
    markValidDetections(detections, frame,degrees)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
videoStream.stop()