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


sam_checkpoint = "../../SAM_ckpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")
sam.eval()

predictor = SamPredictor(sam)

mask_num = 0
save_path = '../../data/sam_labels'
train_pc_path = '../../data/s3dis'
s3dis_frame_path  = '../../data/S2D3D'
bridge_path = '../../data/bridge'
embed_path = '../../data/embeddings'
prompt_path = '../../data/weak_labels'
area_list = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']

for area in area_list:
    os.makedirs(os.path.join(save_path, area), exist_ok=True)
    scene_list = sorted([path.split('/')[-1].rstrip('.pth') for path in glob.glob(os.path.join(train_pc_path, area, '*.pth'))])

    for scene_name in scene_list:
        frame_list = [color_frame.split('/')[-1].rstrip('.pth') for color_frame in glob.glob(os.path.join(embed_path, area,scene_name,'*.pth'))]
        
        if len(frame_list) == 0:
            pcd_data = torch.load(os.path.join(train_pc_path,area,scene_name +'.pth'))
            scene_pcd_np = pcd_data['coord']
            sam_label_pcd = (-1*np.ones((scene_pcd_np.shape[0],1))).astype(np.int32)
            prompt = np.load(os.path.join(prompt_path,area, scene_name+'.npy'))
            for prompt_number in range(prompt.shape[0]):
                if prompt[prompt_number] == 0:
                    continue            
                if pcd_data['semantic_gt'][prompt_number] == -1:
                    continue
                sam_label_pcd[int(prompt_number),0] = int(pcd_data['semantic_gt'][prompt_number])
            print(area ,scene_name)
            np.save(os.path.join(save_path,area, scene_name+'.npy'), sam_label_pcd)
            continue
        
        check_frame = np.array(Image.open(os.path.join(s3dis_frame_path, area,'data','rgb', frame_list[0]+'.png'))) 
        

        sam_predictor = SamPredictor(sam)
        sam_predictor.set_image(check_frame)

        
        pcd_data = torch.load(os.path.join(train_pc_path,area,scene_name +'.pth'))
        scene_pcd_np = pcd_data['coord']
        sam_label_pcd = (-1*np.ones((scene_pcd_np.shape[0],1))).astype(np.int32)
        prompt = np.load(os.path.join(prompt_path,area, scene_name+'.npy'))
        
        mask_dict = dict()
        for frame_number in frame_list:
            
            if not os.path.isfile(os.path.join(embed_path,area,scene_name, frame_number+'.pth')):
                continue
            if not os.path.isfile(os.path.join(bridge_path,area,scene_name, frame_number+'.npy')):
                continue
            frame_embed = torch.load(os.path.join(embed_path,area,scene_name, frame_number+'.pth'))
            frame_bridge = np.load(os.path.join(bridge_path,area,scene_name, frame_number+'.npy'))
            frame_bridge[:,[0,1]] = frame_bridge[:,[1,0]]
            valid_point_list = np.where(frame_bridge[:,2]==1)[0]
            for prompt_number in range(prompt.shape[0]):
                
                if prompt[prompt_number] == 0:
                    continue
                
                if pcd_data['semantic_gt'][prompt_number] == -1:
                    continue
                if frame_bridge[int(prompt_number),2] == 0:
                    continue
                
                sam_predictor.features = frame_embed.cuda()
                sam_input_point = np.expand_dims(np.array([frame_bridge[int(prompt_number),1],frame_bridge[int(prompt_number),0]]).astype(int),0)
                
                masks, scores, logits = sam_predictor.predict(point_coords=sam_input_point,point_labels=np.array([1]))
                sam_label = pcd_data['semantic_gt'][prompt_number]
                for valid_point in valid_point_list:
                    if masks[mask_num, int(frame_bridge[valid_point,0])-1, int(frame_bridge[valid_point,1])-1]:
                        
                        if valid_point in mask_dict:
                            if sam_label.item() in mask_dict[valid_point]:
                                mask_dict[valid_point][sam_label.item()] +=1
                            else:
                                mask_dict[valid_point][sam_label.item()] =1
                        else:
                            mask_dict[valid_point] = dict()
                            mask_dict[valid_point][sam_label.item()] =1
                        
                        sam_label_pcd[valid_point,0] = sorted(mask_dict[valid_point].items(), key=lambda x:x[1], reverse=True)[0][0]                    
                        if len(mask_dict[valid_point]) > 1:  # No voting, drop
                            sam_label_pcd[valid_point,0] = -1
        
        color = np.zeros_like(scene_pcd_np)
        for prompt_number in range(prompt.shape[0]):
            if prompt[prompt_number] == 0:
                continue            
            if pcd_data['semantic_gt'][prompt_number] == -1:
                continue
            sam_label_pcd[int(prompt_number),0] = int(pcd_data['semantic_gt'][prompt_number])
            color[int(prompt[prompt_number]),:] = label_to_colors[sam_label_pcd[prompt_number,0]]
        
        np.save(os.path.join(save_path,area, scene_name+'.npy'), sam_label_pcd)
