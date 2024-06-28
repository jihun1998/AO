import numpy as np
import pickle
import os
import json
from PIL import Image
from matplotlib import pyplot as plt

import glob
import pdb
from segment_anything import sam_model_registry, SamPredictor
import time
import pdb

from ply import *
import torch
import copy

label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0], [255,255,255]]
palette = np.asarray(label_to_colors)

data_3d_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis'
data_2d_root = '/home/vilab/khj/ssd0/pointcept/data/S2D3D'
sampled_cameras_root = '/home/vilab/khj/ssd0/pointcept/data/sampled_cameras'
bridge_root = '/home/vilab/khj/ssd0/pointcept/data/bridge'
weak_labels_root = '/home/vilab/khj/ssd0/pointcept/data/weak_labels'
weak_coords_root = '/home/vilab/khj/ssd0/pointcept/data/weak_coords'
sam_labels_root = '/home/vilab/khj/ssd0/pointcept/data/sam_labels'

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))
area_3d_paths.pop(4)

# predictor = SamPredictor(sam)

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)

    os.makedirs(weak_coords_root+'/'+area_str, exist_ok=True)

    room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))

    area_2d_root = data_2d_root + '/' + area_str + '/data'

    rgb_2d_root = area_2d_root+'/rgb'
    depth_2d_root = area_2d_root+'/depth'
    pose_2d_root = area_2d_root+'/pose'

    rgb_paths = sorted(glob.glob(rgb_2d_root+'/*.png'))
    depth_paths = sorted(glob.glob(depth_2d_root+'/*.png'))
    pose_paths = sorted(glob.glob(pose_2d_root+'/*.json'))
    
    for room_path in room_paths:
        print(room_path)
        room_str = room_path.split('/')[-1][:-4]

        data_3d = torch.load(room_path)

        coord = data_3d['coord']
        label_segment = data_3d['semantic_gt']
        label_instance = data_3d['instance_gt'].reshape(-1)
        label_exist = np.zeros_like(label_instance)
        vote = np.zeros_like(label_segment).repeat(13,1) # (N,13)

        bridge_paths = sorted(glob.glob(bridge_root+'/'+area_str+'/'+room_str+'/*.npy'))
        weak_path = weak_labels_root+'/'+area_str+'/'+room_str+'.npy'
        weak_idx = np.load(weak_path)


        temp = coord[weak_idx==1]
        temp2 = label_segment[weak_idx==1]
        temp3 = label_instance[weak_idx==1]

        temp3 = np.expand_dims(temp3,axis=1)

        temp_all = np.concatenate((temp, temp2, temp3), axis=1)

        np.save(weak_coords_root+'/'+area_str+'/'+room_str+'.npy', temp_all)