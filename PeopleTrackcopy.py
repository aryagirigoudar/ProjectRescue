from functools import cache
import cv2
import numpy as np
from centroidtracker import CentroidTracker
from time import sleep
 

COUNT = 0 
x1 = 0
y1 = 0
x2 = 0

y2 = 0 

PROTOPATH = "/Users/aryagirigoudar/Documents/Python /ProjectRescue/pretrainedModels/MobileNetSSD_deploy.prototxt"
MODELPATH = "/Users/aryagirigoudar/Documents/Python /ProjectRescue/pretrainedModels/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=PROTOPATH, caffeModel=MODELPATH)



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

text = ''
who = set()

def main(frame):
    (H, W) = frame.shape[:2]
    copy = frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)

    person_detections = detector.forward()
    rects = []
    for i in np.arange(0, person_detections.shape[2]):
            # print(person_detections)# -> (1, 1, 100, 7)
            # sleep(1)
        confidence = person_detections[0, 0, i, 2]
            # print(confidence)

        if confidence > 0.70:
            idx = int(person_detections[0, 0, i, 1])

            li = ["cat","dog","person","sheep","cow"]
            if CLASSES[idx] not in li :
                continue
            else:
                who.add(CLASSES[idx])

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            global COUNT
            (startX, startY, endX, endY) = person_box.astype("int")
            rects.append(person_box)
            

        
    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)

    objects = tracker.update(rects)
    
    
    for (objectId, bbox) in objects.items():
        global x1
        global x2
        global y1
        global y2
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        global text 
        
        text = f"Found {who}"
        # .format(objectId+1)

        

    
    cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    # return text,x1,y1,x2,y2


