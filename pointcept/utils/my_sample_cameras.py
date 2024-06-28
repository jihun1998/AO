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


data_3d_root = '/mnt/jihun4/pointceptHJ/data/s3dis'
data_2d_root = '/mnt/jihun4/pointceptHJ/data/S2D3D'
sampled_cameras_root = '/mnt/jihun4/pointceptHJ/data/sampled_cameras'

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))

# area_3d_paths = [area_3d_paths[-1]]


for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)
    
    # sampled_camera_txt = open(sampled_cameras_root+'/'+area_str+'.txt', 'w')

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
        import pdb;pdb.set_trace()
        room_str = room_path.split('/')[-1][:-4]

        for i, rgb_path in enumerate(rgb_paths):

            temp = rgb_path.split('/')[-1].split('_')
            if room_str != temp[2]+'_'+temp[3]:
                continue
            else:
                camera_str = temp[1]
                # sampled_camera_txt.write(room_str + ' ' + camera_str + '\n')
                break
    # sampled_camera_txt.close()
