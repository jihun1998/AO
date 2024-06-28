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

import random

random.seed(4242)

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0]]
palette = np.asarray(label_to_colors)

data_3d_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis'
cls_gt_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis_cls_gt'

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))
area_3d_paths.pop(4)

print(area_3d_paths)

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)

    os.makedirs(cls_gt_root+'/'+area_str, exist_ok=True)
    room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))



    count=0
    for room_path in room_paths:

        print(room_path)
        room_str = room_path.split('/')[-1][:-4]


        data_3d = torch.load(room_path)
        label_segment = data_3d['semantic_gt'].reshape(-1)
        cls_gt = np.unique(label_segment)

        np.save(cls_gt_root+'/'+area_str+'/'+room_str+'.npy', cls_gt)
        