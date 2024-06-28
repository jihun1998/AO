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
data_2d_root = '/home/vilab/khj/ssd0/pointcept/data/S2D3D'
sampled_cameras_root = '/home/vilab/khj/ssd0/pointcept/data/sampled_cameras'
bridge_root = '/home/vilab/khj/ssd0/pointcept/data/bridge'
weak_label_root = '/home/vilab/khj/ssd0/pointcept/data/weak_labels'
used_imgs_root = '/home/vilab/khj/ssd0/pointcept/data/used_imgs'
aa_root = '/home/vilab/khj/ssd0/pointcept/data/align_angle_and_center'

area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))
area_3d_paths.pop(4)

print(area_3d_paths)

for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]
    print(area_str)

    os.makedirs(bridge_root+'/'+area_str, exist_ok=True)
    os.makedirs(weak_label_root+'/'+area_str, exist_ok=True)
    os.makedirs(used_imgs_root+'/'+area_str, exist_ok=True)

    room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))

    area_2d_root = data_2d_root + '/' + area_str + '/data'

    rgb_2d_root = area_2d_root+'/rgb'
    depth_2d_root = area_2d_root+'/depth'
    pose_2d_root = area_2d_root+'/pose'

    rgb_paths = sorted(glob.glob(rgb_2d_root+'/*.png'))
    depth_paths = sorted(glob.glob(depth_2d_root+'/*.png'))
    pose_paths = sorted(glob.glob(pose_2d_root+'/*.json'))

    temp = list(zip(rgb_paths, depth_paths, pose_paths))
    random.shuffle(temp)
    rgb_paths, depth_paths, pose_paths = zip(*temp)

    aa_txt_path = aa_root + '/' + area_str + '.txt'
    aa_txt = open(aa_txt_path, 'r')
    aa_list = aa_txt.readlines()

    aa_dict = {}
    center_dict = {}
    for aaidx, aa in enumerate(aa_list):
        temp = aa.split(' ')
        aa_dict[temp[0]] = int(temp[1])
        center_dict[temp[0]] = np.array([float(temp[2]), float(temp[3]), float(temp[4][:-1])])

    count=0
    for room_path in room_paths:

        print(room_path)
        room_str = room_path.split('/')[-1][:-4]

        os.makedirs(bridge_root+'/'+area_str+'/'+room_str, exist_ok=True)

        used_imgs_txt = open(used_imgs_root+'/'+area_str+'/'+room_str+'.txt', 'w')

        data_3d = torch.load(room_path)

        coord = data_3d['coord']
        color = data_3d['color']
        label_segment = data_3d['semantic_gt'].reshape(-1)
        label_instance = data_3d['instance_gt'].reshape(-1)
        viewable_all = np.zeros_like(label_segment)

        ##
aa_txt = open(aa_txt_path, 'r')
aa_list = aa_txt.readlines()

for aaidx, aa in enumerate(aa_list):
    temp = aa.split(' ')
    aa_dict[temp[0]] = int(temp[1])
    center_dict[temp[0]] = np.array([float(temp[2]), float(temp[3]), float(temp[4][:-1])])
room_center = center_dict[room_str]
angle = 360-aa_dict[room_str]
angle = (2 - angle / 180) * np.pi
rot_cos, rot_sin = np.cos(angle), np.sin(angle)
rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
coord -= room_center
coord = coord @ np.transpose(rot_t)
coord += room_center
        ##

        count = 0
        for i, rgb_path in enumerate(rgb_paths):

            if count==50:
                break

            temp = rgb_path.split('/')[-1].split('_')
            if room_str != temp[2]+'_'+temp[3]:
                continue

            count += 1
            print(rgb_path)
            used_imgs_txt.write(rgb_path+'\n')

            rgb_str = rgb_path.split('/')[-1][:-4]
            depth_path = depth_paths[i]
            pose_path = pose_paths[i]

            rgb = np.array(Image.open(rgb_path))
            depth = np.array(Image.open(depth_path))/512
            with open(pose_path, 'r') as pp:
                pose = json.load(pp)

            k_matrix = np.array(pose['camera_k_matrix']) 
            rt_matrix = np.array(pose['camera_rt_matrix'])
            height = k_matrix[0,2]*2 - 1
            width = k_matrix[1,2]*2 - 1

            
            coord_homogenous = np.concatenate((coord, np.ones((coord.shape[0],1))),1) # (N,4)
            coord_camera = np.transpose(np.matmul(np.concatenate((rt_matrix,np.array([[0,0,0,1]]))),np.transpose(coord_homogenous)))
            coord_img = np.transpose(np.matmul(np.matmul(k_matrix, rt_matrix),np.transpose(coord_homogenous)))
            
            coord_img = np.round(coord_img/np.expand_dims(coord_img[:,2],1))
            
            valid_idx = np.where((coord_img[:,0]>0)*(coord_img[:,1]>0)*(coord_img[:,0]<height)*(coord_img[:,1]<width))[0]

            if valid_idx.shape[0]!=0:
                valid_coord = coord_img[valid_idx,:2].astype(np.uint16) # (V,2)
                valid_depth_gt = depth[valid_coord[:,1], valid_coord[:,0]]
                valid_depth_pred = coord_camera[valid_idx,2]

                viewable_idx_ = np.abs(valid_depth_gt - valid_depth_pred)<0.1
                viewable_idx = valid_idx[viewable_idx_]

                if viewable_idx.shape[0]>0:

                    viewable_coord = coord_img[viewable_idx].astype(np.uint16)
                                        
                    bridge = np.zeros_like(coord)
                    
                    bridge[viewable_idx] = viewable_coord
                    bridge = bridge.astype(np.uint16)

                    viewable_all[viewable_idx] = 1
                    print(np.unique(label_instance[viewable_idx]))
                    
                    np.save(bridge_root+'/'+area_str+'/'+room_str+'/'+rgb_str+'.npy', bridge)

                    # write_ply('./temp.ply',  [coord[viewable_idx], color[viewable_idx]], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    # plt.imsave('./temp.png', plt.imread(rgb_path))

                    # pdb.set_trace()
                    # rgb[viewable_coord[:,1], viewable_coord[:,0],:] = palette[label_segment[viewable_idx]]
                    # temp = viewable_coord[:,1]+1
                    # temp[temp>height] = height
                    # rgb[temp, viewable_coord[:,0],:] = palette[label_segment[viewable_idx]]
                    # temp = viewable_coord[:,1]-1
                    # temp[temp<0] = 0
                    # rgb[temp, viewable_coord[:,0],:] = palette[label_segment[viewable_idx]]
                    # temp = viewable_coord[:,0]+1
                    # temp[temp>height] = width
                    # rgb[viewable_coord[:,1], temp,:] = palette[label_segment[viewable_idx]]
                    # temp = viewable_coord[:,0]-1
                    # temp[temp<0] = 0
                    # rgb[viewable_coord[:,1], temp,:] = palette[label_segment[viewable_idx]]

                    # plt.imsave('./temp_gt.png', rgb)

                    # pdb.set_trace()
                
                else:
                    print('pass')

        used_imgs_txt.close()
        viewable_instance = label_instance[viewable_all==1]
        viewable_segment = label_segment[viewable_all==1]
        weak_mask = np.zeros_like(label_segment).reshape(-1)

        weak_instance = []
        print('viewable')
        for iidx in np.unique(viewable_instance):
            weak_instance.append(iidx)
            idx_instance = np.where(viewable_instance==iidx)[0]
            cidx = label_segment[idx_instance][0]
            print(iidx, CATS[cidx])
            idx_weak = idx_instance[idx_instance.shape[0]//2]
            temp_idx = np.arange(len(weak_mask))
            weak_mask[temp_idx[viewable_all==1][idx_weak]] = 1

        print('ocludded')
        for iidx in np.unique(label_instance):
            if iidx not in weak_instance:
                idx_instance = np.where(label_instance==iidx)[0]
                cidx = label_segment[idx_instance][0]
                print(iidx, CATS[cidx])
                idx_weak = idx_instance[idx_instance.shape[0]//2]
                weak_mask[idx_weak] = 1

        np.save(weak_label_root+'/'+area_str+'/'+room_str+'.npy', weak_mask)

        # pdb.set_trace()

        # write_ply('./temp.ply',  [coord[viewable_all==1], color[viewable_all==1]], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply('./temp2.ply',  [coord,palette[label_segment]*1.], ['x', 'y', 'z', 'red', 'green', 'blue'])