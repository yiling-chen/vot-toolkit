from __future__ import print_function
import sys
import cv2
import json
import torch
import random
import numpy as np
from glob import glob
from os import walk
from os.path import realpath, dirname, join, exists

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from vot import parse_region

sequence_root_folder = 'sequences/sequence-4'
prefix = 'vlc-record-2018-08-29-11h49m34s-108NE-MAIN-2018-08-28-1538-1638.mp4'

files = glob(join(sequence_root_folder, '*.json'))
print(files[0])

with open(files[0]) as f:
     data = f.read()

json_obj = json.loads(data)

# read and cache all frames
image_list = glob(join(sequence_root_folder, '*.jpg'))
image_list.sort()

src_images = []
frames = []
for f in image_list:
    src_images.append(cv2.imread(f))
    frames.append(cv2.imread(f))

height, width = frames[0].shape[:2]

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

for obj_id in range(len(json_obj['annotations'][0]['data']['objects'])):
    # get the first frame
    frm = json_obj['annotations'][0]['data']['objects'][obj_id]['frames'][0]
    x1 = int(frm['boundingBox']['x1'] * width)
    y1 = int(frm['boundingBox']['y1'] * height)
    x2 = int(frm['boundingBox']['x2'] * width)
    y2 = int(frm['boundingBox']['y2'] * height)

    frm_idx = frm['frameNumber']
    cv2.rectangle(frames[frm_idx], (x1, y1), (x2, y2), (0,0,255), 2)

    target_pos, target_sz = np.array([(x1+x2)/2, (y1+y2)/2]), np.array([x2-x1, y2-y1])
    state = SiamRPN_init(src_images[frm_idx], target_pos, target_sz, net)  # init tracker

    color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 128))
    num_frames = len(json_obj['annotations'][0]['data']['objects'][obj_id]['frames'])
    for j in range(1, num_frames):
        frm = json_obj['annotations'][0]['data']['objects'][obj_id]['frames'][j]

        x1 = int(frm['boundingBox']['x1'] * width)
        y1 = int(frm['boundingBox']['y1'] * height)
        x2 = int(frm['boundingBox']['x2'] * width)
        y2 = int(frm['boundingBox']['y2'] * height)

        frm_idx = frm['frameNumber']
        cv2.rectangle(frames[frm_idx], (x1, y1), (x2, y2), (0,0,255), 2)

        # Start timer
        timer = cv2.getTickCount()

        state = SiamRPN_track(state, src_images[frm_idx])  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(x) for x in res]

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.rectangle(frames[frm_idx], (res[0], res[1]), (res[0]+res[2], res[1]+res[3]), color, 2)

        # Display tracker type on frame
        cv2.putText(frames[frm_idx], "SiamRPN", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)

        # Display FPS on frame
        cv2.putText(frames[frm_idx], "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)


# write all frames to video
writer = cv2.VideoWriter(sequence_root_folder + '-SiamRPN.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, (width, height))
if not writer.isOpened():
    print('Failed to open video')
    exit(1)

for frm in frames:
    writer.write(frm)

writer.release()

