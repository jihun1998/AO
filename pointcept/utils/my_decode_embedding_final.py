import numpy as np
import pickle
import os
import json
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import glob
import pdb
from segment_anything import sam_model_registry, SamPredictor # original sam
# from mobile_sam import SamPredictor, sam_model_registry  # mobilesam
import time
import pdb
import torch
import open3d as o3d

start = time.time()
sam_checkpoint = "../../SAM_ckpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")

predictor = SamPredictor(sam)

sam_embed_dict = dict()
camera_list = []
area_num = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']

data_path = '../../used_imgs'
aa_root = '../../data/align_angle_and_center'
rgb_path= 'data/rgb'


for area in area_num:
    
    
    area_path = os.path.join(data_path, area, '*.txt')
    room_list = glob.glob(area_path)
    sam_embed_dict = dict()
    camera_list = []
    aa_txt_path = aa_root + '/' + area + '.txt'
    aa_txt = open(aa_txt_path, 'r')
    aa_list = aa_txt.readlines()
    aa_dict = {}
    center_dict = {}
    for aaidx, aa in enumerate(aa_list):
        temp = aa.split(' ')
        aa_dict[temp[0]] = int(temp[1])
        center_dict[temp[0]] = np.array([float(temp[2]), float(temp[3]), float(temp[4][:-1])])

    for room in room_list:
        
        room_file = open(room, 'r')
        img_list = room_file.readlines()
        room_name = room.split('/')[-1].rstrip('.txt')
        
        os.makedirs(os.path.join('../../data/embeddings',area,room_name), exist_ok=True)
        
        data = torch.load(os.path.join('../../data/s3dis', area,room_name + '.pth' ))
        point_cloud = data['coord']

        room_center = center_dict[room_name]
        angle = 360-aa_dict[room_name]
        angle = (2 - angle / 180) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        point_cloud -= room_center
        point_cloud = point_cloud @ np.transpose(rot_t)
        point_cloud += room_center

        color = data['color']
        for img in img_list:
            pose_path = os.path.join('../../data/S2D3D/Area_'+area[-1], 'data/pose', img.split('/')[-1].replace('rgb.png\n','pose.json'))
            img_name= img.split('/')[-1].rstrip('.png\n')
            with open(pose_path, 'r') as pose_json:
                pose = json.load(pose_json)
            k_matrix = np.array(pose['camera_k_matrix']) 
            rt_matrix = np.array(pose['camera_rt_matrix'])
            real_coord = np.concatenate((point_cloud, np.ones((point_cloud.shape[0],1))),1)
            
            camera_coord = np.transpose(np.matmul(np.concatenate((rt_matrix,np.array([[0,0,0,1]]))),np.transpose(real_coord)))
        
            image_coord = np.transpose(np.matmul(np.matmul(k_matrix, rt_matrix),np.transpose(real_coord)))
            image_coord = np.round(image_coord/np.expand_dims(image_coord[:,2],1))
            
            rgb = np.zeros((1100,1100,3))
            bridge = np.zeros_like(image_coord)
            frame_valid = np.where((image_coord[:,0]>0) * (image_coord[:,1]>0) * (image_coord[:,0]<=1080) * (image_coord[:,1]<=1080) * (camera_coord[:,2]>0))[0]
            if frame_valid.shape[0] < 3:
                img = Image.fromarray(rgb[0:1080,0:1080,:].astype(np.uint8))
                predictor.set_image(rgb[0:1080,0:1080,:].astype(np.uint8))
                torch.save(predictor.features.cpu(), os.path.join('../../data/embeddings',area,room_name,img_name+'.pth'))
                continue
            
            point_cloud_inframe = point_cloud[frame_valid,:]
            
            r = rt_matrix[:,0:3]
            t = rt_matrix[:,3]
            camera = - np.matmul(np.transpose(r), t)
            
            
            
            img = np.asarray(Image.open(os.path.join('../../data/S2D3D', area,'data','rgb', img_name+'.png')))

            predictor.set_image(img)
            torch.save(predictor.features.cpu(), os.path.join('../../data/embeddings',area,room_name,img_name+'.pth'))
            

