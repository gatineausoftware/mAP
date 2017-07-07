
from __future__ import division
import numpy as np




def recallThreshold(detection_number, bin_number, total_bins, num_gt):
    if detection_number >= (bin_number / total_bins * num_gt):
        return True
    else:
        return False



def getThreshold(detection_thresholds, num_gt, num_bins):

    detection_thresholds.sort()
    num_detections = len(detection_thresholds)
    bin = 1
    thresholds = {}

    for index in range(0, num_detections):
            if recallThreshold(index+1, bin, num_bins, num_gt):
                thresholds[bin/num_bins] = detection_thresholds[index]
                bin+=1

    return thresholds



class BoundingBox:

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def x_overlap(self, other):

        left = max(self.xmin, other.xmin)
        right = min(self.xmax, other.xmax)

        if left < right:
            return right - left
        else:
            return 0

    def y_overlap(self, other):
        top = min(self.ymax, other.ymax)
        bottom = max(self.ymin, other.ymin)

        if top > bottom:
            return top - bottom
        else:
            return 0

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


    def intersection(self, other):
        return self.x_overlap(self, other) * self.y_overlap(self, other)

    def union(self, other):
        return self.area() + other.area() - self.intersection(self, other)

    def iou(self, other):
        return self.intersection(self, other) / self.union(self, other)


class Detection:

    def __init__(self, confidence, xmin, xmax, ymin, ymax, class_type):
        self.confidence = confidence
        self.boundingBox = BoundingBox(xmin, xmax, ymin, ymax)
        self.class_type = class_type
        self.matched = False

    def __init__(self, confidence):
        self.confidence = confidence

    def __str__(self):
        return str(self.confidence)



def initializeDetections(detections):
    a = []
    for i in detections:
        a.append(Detection(i))

    return a






a = initializeDetections([0.1, 0.1, 0.2, 0.2])

