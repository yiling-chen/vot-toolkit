import vot
import sys
import time
import cv2
import numpy
import collections

class CSRTTracker(object):

    def __init__(self, image, region):
        self.tracker = cv2.TrackerCSRT_create()

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.tracker.init(image, (left, top, right - left, bottom - top))

    def track(self, image):
        ok, bbox = self.tracker.update(image)
        if ok:
            val = 0.5
        else:
            val = 0.05

        return vot.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3]), val

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = CSRTTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region, confidence = tracker.track(image)
    handle.report(region, confidence)

