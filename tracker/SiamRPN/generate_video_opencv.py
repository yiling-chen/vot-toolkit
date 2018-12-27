from __future__ import print_function
import sys
import cv2  # imread
import torch
import numpy as np
from glob import glob
from os import walk
from os.path import realpath, dirname, join, exists

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from vot import parse_region

major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]


def process_video(groundtruth_path, image_path, out_video):
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

    print('processing sequence', out_video)
    with open(groundtruth_path) as f:
        groundtruth = f.readlines()

    groundtruth = [x.rstrip() for x in groundtruth]

    image_filenames = [y for x in walk(image_path) for y in glob(join(x[0], '*.jpg'))]
    image_filenames.sort()

    assert len(image_filenames) == len(groundtruth)

    image = cv2.imread(image_filenames[0])
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('X','V','I','D'), 15, (width , height))

    if not writer.isOpened():
        print('Failed to open video')
        return

    # VOT sequence
    # polygon_ = parse_region(groundtruth[0])
    # cx, cy, w, h = get_axis_aligned_bbox(polygon_)
    # target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

    polygon = [float(x) for x in groundtruth[0].split(',')]
    ok = tracker.init(image, (polygon[0], polygon[1], polygon[2], polygon[3]))

    for i in range(len(image_filenames)):
        image = cv2.imread(image_filenames[i])
        polygon = [float(x) for x in groundtruth[i].split(',')]
        polygon = [int(x) for x in polygon]
        
        # VOT sequence
        # cv2.line(image, (polygon[0], polygon[1]), (polygon[2], polygon[3]), (0, 0, 255), 2)
        # cv2.line(image, (polygon[2], polygon[3]), (polygon[4], polygon[5]), (0, 0, 255), 2)
        # cv2.line(image, (polygon[4], polygon[5]), (polygon[6], polygon[7]), (0, 0, 255), 2)
        # cv2.line(image, (polygon[6], polygon[7]), (polygon[0], polygon[1]), (0, 0, 255), 2)

        cv2.rectangle(image, (polygon[0], polygon[1]), (polygon[0]+polygon[2], polygon[1]+polygon[3]), (0, 0, 255), 2)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(image)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(image, "Tracking failure detected", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,255),2)

        # Display tracker type on frame
        cv2.putText(image, tracker_type + " Tracker", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)
        
        # Display FPS on frame
        cv2.putText(image, "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)

        writer.write(image)

    writer.release()


sequence_root_folder = '../../workspace-vot2018/sequences/'

with open(join(sequence_root_folder, 'list.txt')) as f:
    seq_list = f.readlines()

seq_list = [x.rstrip() for x in seq_list]

for sequence in seq_list:
    ground_truth = join(join(sequence_root_folder, sequence), 'groundtruth.txt')
    image_folder = join(join(sequence_root_folder, sequence), 'color')
    process_video(ground_truth, image_folder, './videos/'+sequence+'.avi')
