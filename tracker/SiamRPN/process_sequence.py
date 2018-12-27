from __future__ import print_function
import sys
import cv2
import json
import torch
import numpy as np
from glob import glob
from os import walk
from os.path import realpath, dirname, join, exists

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from vot import parse_region

sequence_root_folder = 'sequences/sequence-2'
prefix = 'vlc-record-2018-08-29-11h49m34s-108NE-MAIN-2018-08-28-1538-1638.mp4'

files = glob(join(sequence_root_folder, '*.json'))
print(files[0])

with open(files[0]) as f:
     data = f.read()

json_obj = json.loads(data)

obj_id = 0

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


writer = cv2.VideoWriter(sequence_root_folder + '-SiamRPN.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, (width, height))
if not writer.isOpened():
    print('Failed to open video')
    exit(1)

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())


target_pos, target_sz = np.array([(x1+x2)/2, (y1+y2)/2]), np.array([x2-x1, y2-y1])
state = SiamRPN_init(img, target_pos, target_sz, net)  # init tracker

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

    state = SiamRPN_track(state, img)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(x) for x in res]

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.rectangle(img, (res[0], res[1]), (res[0]+res[2], res[1]+res[3]), (255,0,0), 2)

    # Display tracker type on frame
    cv2.putText(img, "SiamRPN", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)

    # Display FPS on frame
    cv2.putText(img, "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)

    writer.write(img)

writer.release()
