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

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0], [255,255,255]]
palette = np.asarray(label_to_colors)

data_3d_root = '../../data/s3dis'
data_2d_root = '../../data/S2D3D'
bridge_root = '../../data/bridge'
weak_root = '../../data/weak_labels'

os.makedirs(weak_root, exist_ok=True)

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]

    os.makedirs(weak_root+'/'+area_str, exist_ok=True)

    room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))[7:]

    area_2d_root = data_2d_root + '/' + area_str + '/data'

    rgb_2d_root = area_2d_root+'/rgb'
    depth_2d_root = area_2d_root+'/depth'
    pose_2d_root = area_2d_root+'/pose'

    rgb_paths = sorted(glob.glob(rgb_2d_root+'/*.png'))
    depth_paths = sorted(glob.glob(depth_2d_root+'/*.png'))
    pose_paths = sorted(glob.glob(pose_2d_root+'/*.json'))
    
    for room_path in room_paths:
        room_str = room_path.split('/')[-1][:-4]

        data_3d = torch.load(room_path)

        coord = data_3d['coord']
        color = data_3d['color']
        label_segment = data_3d['semantic_gt'].reshape(-1)
        label_instance = data_3d['instance_gt'].reshape(-1)

        viewable_all = np.zeros_like(label_instance)
        weak_mask = np.zeros_like(label_instance)

        bridge_paths = sorted(glob.glob(bridge_root+'/'+area_str+'/'+room_str+'/*.npy'))

        for bridge_path in bridge_paths:
            
            bridge = np.load(bridge_path)
            viewable_idx = bridge[:,2]==1
            viewable_all[viewable_idx] = 1
        
        
        viewable_instance = label_instance[viewable_all==1]
        viewable_segment = label_segment[viewable_all==1]

        weak_instance = []
        for iidx in np.unique(viewable_instance):
            weak_instance.append(iidx)
            idx_instance = np.where(viewable_instance==iidx)[0]
            cidx = label_segment[idx_instance][0]
            idx_weak = idx_instance[idx_instance.shape[0]//2]
            temp_idx = np.arange(len(weak_mask))
            weak_mask[temp_idx[viewable_all==1][idx_weak]] = 1

        for iidx in np.unique(label_instance):
            if iidx not in weak_instance:
                idx_instance = np.where(label_instance==iidx)[0]
                cidx = label_segment[idx_instance][0]
                idx_weak = idx_instance[idx_instance.shape[0]//2]
                weak_mask[idx_weak] = 1
                
        np.save(weak_root+'/'+area_str+'/'+room_str+'.npy', weak_mask)

