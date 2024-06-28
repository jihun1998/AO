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

data_3d_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis'
data_2d_root = '/home/vilab/khj/ssd0/pointcept/data/S2D3D'
sampled_cameras_root = '/home/vilab/khj/ssd0/pointcept/data/sampled_cameras'
bridge_root = '/home/vilab/khj/ssd0/pointcept/data/bridge'
weak_root = '/home/vilab/khj/ssd0/pointcept/data/weak_labels_0.02'

os.makedirs(weak_root, exist_ok=True)

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))

num_all_point = 0
num_weak_point = 0

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)

    os.makedirs(weak_root+'/'+area_str, exist_ok=True)

    room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))
    
    for room_path in room_paths:
        print(room_path)
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
            # print(viewable_all.sum())
        

        viewable_instance = label_instance[viewable_all==1]
        viewable_segment = label_segment[viewable_all==1]

        weak_instance = []
        print('viewable')
        for iidx in np.unique(viewable_instance):
            weak_instance.append(iidx)
            idx_instance = np.where(viewable_instance==iidx)[0]
            cidx = label_segment[idx_instance][0]
            print(iidx, CATS[cidx])
            #
            num_ins = idx_instance.shape[0]//6
            sampling_array = (np.arange(5)+1)*num_ins
            idx_weak = idx_instance[sampling_array]
            #
            temp_idx = np.arange(len(weak_mask))
            weak_mask[temp_idx[viewable_all==1][idx_weak]] = 1

        print('ocludded')
        for iidx in np.unique(label_instance):
            if iidx not in weak_instance:
                idx_instance = np.where(label_instance==iidx)[0]
                cidx = label_segment[idx_instance][0]
                print(iidx, CATS[cidx])
                #
                num_ins = idx_instance.shape[0]//6
                sampling_array = (np.arange(5)+1)*num_ins
                idx_weak = idx_instance[sampling_array]
                #
                weak_mask[idx_weak] = 1
        
        np.save(weak_root+'/'+area_str+'/'+room_str+'.npy', weak_mask)
        num_all_point += weak_mask.shape[0]
        num_weak_point += weak_mask.sum()
        print(100*num_weak_point/num_all_point)

        
        # pdb.set_trace()
        # # color = color*0
        # # color[weak_mask==1,0] = 255
        # # write_ply('./temp.ply',  [coord, color.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

        # pdb.set_trace()
