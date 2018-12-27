from __future__ import print_function
import sys
import cv2
import json
import torch
import numpy as np
from glob import glob
from os import walk
from os.path import realpath, dirname, join, exists

major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

sequence_root_folder = 'sequences/sequence-1'
prefix = 'vlc-record-2018-08-29-11h48m10s-108NE-MAIN-2018-08-28-1538-1638.mp4'

files = glob(join(sequence_root_folder, '*.json'))
print(files[0])

with open(files[0]) as f:
     data = f.read()

json_obj = json.loads(data)

obj_id = 3

# get the first frame
frm = json_obj['annotations'][0]['data']['objects'][obj_id]['frames'][0]
fname = join(sequence_root_folder, prefix + '--{:03d}.jpg'.format(frm['frameNumber']))
print(fname)
if not exists(fname):
    print('Error: file not exists!')
img = cv2.imread(fname)
height, width = img.shape[:2]
x1 = int(frm['boundingBox']['x1'] * width)
y1 = int(frm['boundingBox']['y1'] * height)
x2 = int(frm['boundingBox']['x2'] * width)
y2 = int(frm['boundingBox']['y2'] * height)


writer = cv2.VideoWriter(sequence_root_folder + '-' + tracker_type + '.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, (width, height))
if not writer.isOpened():
    print('Failed to open video')
    exit(1)

ok = tracker.init(img, (x1, y1, x2-x1, y2-y1))


for frm in json_obj['annotations'][0]['data']['objects'][obj_id]['frames']:
    fname = join(sequence_root_folder, prefix + '--{:03d}.jpg'.format(frm['frameNumber']))
    if not exists(fname):
        print('Error: file not exists!')
    img = cv2.imread(fname)

    # annotation
    x1 = int(frm['boundingBox']['x1'] * width)
    y1 = int(frm['boundingBox']['y1'] * height)
    x2 = int(frm['boundingBox']['x2'] * width)
    y2 = int(frm['boundingBox']['y2'] * height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(img)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,255),2)

    # Display tracker type on frame
    cv2.putText(img, tracker_type + " Tracker", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)
    
    # Display FPS on frame
    cv2.putText(img, "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)

    writer.write(img)

writer.release()
