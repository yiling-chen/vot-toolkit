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

def process_video(net, groundtruth_path, image_path, out_video):
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
    target_pos, target_sz = np.array([polygon[0]+polygon[2]/2, polygon[1]+polygon[3]/2]), np.array([polygon[2], polygon[3]])
    state = SiamRPN_init(image, target_pos, target_sz, net)  # init tracker

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

        state = SiamRPN_track(state, image)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(x) for x in res]

        cv2.rectangle(image, (res[0], res[1]), (res[0]+res[2], res[1]+res[3]), (255,0,0), 2)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.rectangle(image, (res[0], res[1]), (res[0]+res[2], res[1]+res[3]), (255,0,0), 2)

        # Display tracker type on frame
        cv2.putText(image, "SiamRPN", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)

        # Display FPS on frame
        cv2.putText(image, "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)

        writer.write(image)

    writer.release()


# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

sequence_root_folder = '../../workspace-vot2018/sequences/'

with open(join(sequence_root_folder, 'list.txt')) as f:
    seq_list = f.readlines()

seq_list = [x.rstrip() for x in seq_list]

for sequence in seq_list:
    ground_truth = join(join(sequence_root_folder, sequence), 'groundtruth.txt')
    image_folder = join(join(sequence_root_folder, sequence), 'color')
    process_video(net, ground_truth, image_folder, './videos/'+sequence+'.avi')
