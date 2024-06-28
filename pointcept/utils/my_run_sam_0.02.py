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
weak_labels_root = '/home/vilab/khj/ssd0/pointcept/data/weak_labels_0.02'
sam_labels_root = '/home/vilab/khj/ssd0/pointcept/data/sam_labels_0.02'

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))
area_3d_paths.pop(4)

area_3d_paths = area_3d_paths[3:]
print(area_3d_paths)
sam_checkpoint = "/home/vilab/khj/ssd0/pointcept/pretrained/sam_vit_h.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")
sam.eval()

predictor = SamPredictor(sam)

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)

    os.makedirs(sam_labels_root+'/'+area_str, exist_ok=True)

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

        os.makedirs(bridge_root+'/'+area_str+'/'+room_str, exist_ok=True)

        data_3d = torch.load(room_path)

        coord = data_3d['coord']
        label_segment = data_3d['semantic_gt']
        label_instance = data_3d['instance_gt'].reshape(-1)
        label_exist = np.zeros_like(label_instance)
        
        vote = np.zeros_like(label_segment).repeat(13,1) # (N,13)

        bridge_paths = sorted(glob.glob(bridge_root+'/'+area_str+'/'+room_str+'/*.npy'))
        weak_path = weak_labels_root+'/'+area_str+'/'+room_str+'.npy'
        weak_idx = np.load(weak_path)

        for bridge_path in bridge_paths:
            
            bridge = np.load(bridge_path)
            
            rgb_path = rgb_2d_root + '/' + bridge_path.split('/')[-1][:-4] + '.png'
            rgb = np.array(Image.open(rgb_path))

            height = rgb.shape[0]
            width = rgb.shape[1]

            viewable_idx = bridge[:,2]==1
            viewable_coord = bridge[viewable_idx,:2]
            viewable_segment = label_segment[viewable_idx].reshape(-1) # (V,)

            viewable_weak_idx = np.where(weak_idx[viewable_idx]==1)[0]
            viewable_weak_coord = copy.deepcopy(viewable_coord[viewable_weak_idx]) # (V,2)
            viewable_weak_segment = viewable_segment[viewable_weak_idx]

            print(viewable_weak_idx.shape[0])

            if viewable_weak_coord.shape[0]!=0:
                
                predictor.set_image(rgb)

                viewable_weak_coord = viewable_weak_coord.astype(np.float32)
                input_points = torch.from_numpy(viewable_weak_coord).unsqueeze(1).cuda() # (V,1,2)
                input_points = input_points * 1024 / 1080
                input_labels = torch.ones_like(input_points[:,:,0])

                masks, scores, logits = predictor.predict_torch(input_points, input_labels, multimask_output=True)
                masks = masks.cpu().detach().numpy()

                for midx, m in enumerate(masks):

                    cls_now = viewable_weak_segment[midx]

                    mask_now = m[0]
                    sam_expanded_regions = mask_now[viewable_coord[:,1], viewable_coord[:,0]]
                    temp_idx = np.arange(len(viewable_idx))
                    vote[temp_idx[viewable_idx][sam_expanded_regions],cls_now] += 1

                    # mask_now = m[1]
                    # sam_expanded_regions = mask_now[viewable_coord[:,1], viewable_coord[:,0]]
                    # temp_idx = np.arange(len(viewable_idx))
                    # vote[temp_idx[viewable_idx][sam_expanded_regions],1,cls_now] += 1

                    # mask_now = m[2]
                    # sam_expanded_regions = mask_now[viewable_coord[:,1], viewable_coord[:,0]]
                    # temp_idx = np.arange(len(viewable_idx))
                    # vote[temp_idx[viewable_idx][sam_expanded_regions],2,cls_now] += 1

                # masks, scores, logits = predictor.predict_torch(input_points, input_labels)
                # masks = masks.cpu().detach().numpy()

                # for midx, m in enumerate(masks):
                #     cls_now = viewable_weak_segment[midx]

                #     mask_now = m[0]
                #     sam_expanded_regions = mask_now[viewable_coord[:,1], viewable_coord[:,0]]
                #     temp_idx = np.arange(len(viewable_idx))
                #     vote[temp_idx[viewable_idx][sam_expanded_regions],3,cls_now] += 1
        
        # sam_result = np.argmax(vote, axis=1)
        # sam_result[np.sum(vote,axis=1)==0] = -1
        # sam_result = np.expand_dims(sam_result, axis=1)
        # data_3d['semantic_gt']

        # sam_result[weak_idx==1] = label_segment[weak_idx==1]
        # np.save(sam_labels_root+'/'+area_str+'/'+room_str+'.npy', sam_result)

        sam_result = np.argmax(vote, axis=1)
        sam_result[np.sum(vote,axis=1)==0] = -1
        sam_result = np.expand_dims(sam_result, axis=1)

        np.save(sam_labels_root+'/'+area_str+'/'+room_str+'.npy', sam_result)
        # write_ply('./temp.ply',  [coord, palette[sam_label].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])