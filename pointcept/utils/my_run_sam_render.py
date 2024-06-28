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


sam_checkpoint = "/mnt/jihun2/SAM_ckpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")
sam.eval()

predictor = SamPredictor(sam)

mask_num = 0
# save_path = '/mnt/jihun4/pointceptHJ/data/rendered_sam_labels2'
save_path = '/mnt/jihun4/pointceptHJ/data/sam_labels_mobile'
scannet_pc_path = '/mnt/jihun3/s3dis_cpcm'
train_pc_path = '/mnt/jihun4/pointceptHJ/data/s3dis'
# scannet_frame_path = '/mnt/jihun2/ScanNetv2/frames'
# scannet_frame_path  = '/mnt/jihun4/pointceptHJ/data/rendered_image'
scannet_frame_path  = '/mnt/jihun4/pointceptHJ/data/S2D3D'
# scannet_rendering_path = '/mnt/jihun2/ScanNetv2/rendering_results'
# bridge_path = '/mnt/jihun4/pointceptHJ/data/rendered_bridge'
bridge_path = '/mnt/jihun4/pointceptHJ/data/bridge'
# bridge_path2 = '/mnt/jihun4/pointceptHJ/data/bridge2'
# embed_path = '/mnt/jihun4/pointceptHJ/data/rendered_embeddings'
embed_path = '/mnt/jihun4/pointceptHJ/data/embeddings_mobile'
# embed_path2 = '/mnt/jihun4/pointceptHJ/data/embeddings2'
prompt_path = '/mnt/jihun4/pointceptHJ/data/weak_labels'
scene_pfix = '_vh_clean_2.ply'
# scene_list = glob.glob(os.path.join(scannet_pc_path,'/scene*'))
area_list = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
# area_list = [ 'Area_2', 'Area_3', 'Area_4', 'Area_6']

# import pdb;pdb.set_trace()
# prompt_eff = torch.load('/mnt/jihun3/pointcept/data/scannet/points20')
for area in area_list:
    os.makedirs(os.path.join(save_path, area), exist_ok=True)
    scene_list = sorted([path.split('/')[-1].rstrip('.pth') for path in glob.glob(os.path.join(train_pc_path, area, '*.pth'))])

    for scene_name in scene_list:
        # scene_name = 'conferenceRoom_1'
        frame_list = [color_frame.split('/')[-1].rstrip('.pth') for color_frame in glob.glob(os.path.join(embed_path, area,scene_name,'*.pth'))]
        # frame_list2 = [color_frame2.split('/')[-1].rstrip('.pth') for color_frame2 in glob.glob(os.path.join(embed_path2, area,scene_name,'*.pth'))]
        
        # import pdb;pdb.set_trace()
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
        
        # check_frame = np.array(Image.open(os.path.join(scannet_frame_path, area, frame_list[0]+'.jpg'))) 
        check_frame = np.array(Image.open(os.path.join(scannet_frame_path, area,'data','rgb', frame_list[0]+'.png'))) 
        
        # if check_frame.shape[0] == 968 and check_frame.shape[1] == 1296:
            # continue

        sam_predictor = SamPredictor(sam)
        sam_predictor.set_image(check_frame)

        # scene_pcd = o3d.io.read_point_cloud(os.path.join(scannet_pc_path, scene_name, scene_name+scene_pfix))
        # scene_pcd_np = np.asarray(scene_pcd.points)

        # data = torch.load(os.path.join(scannet_pc_path, scene_name))
        # import pdb;pdb.set_trace()
        # scene_pcd_np = torch.load('/mnt/jihun2/pointceptScanNet/train/scene0000_00.pth')['coord']
        # os.makedirs(os.path.join(save_path, scene_name), exist_ok=True)
        pcd_data = torch.load(os.path.join(train_pc_path,area,scene_name +'.pth'))
        scene_pcd_np = pcd_data['coord']
        sam_label_pcd = (-1*np.ones((scene_pcd_np.shape[0],1))).astype(np.int32)
        prompt = np.load(os.path.join(prompt_path,area, scene_name+'.npy'))
        # import pdb;pdb.set_trace()
        # prompt = prompt_eff[scene_name]
        mask_dict = dict()
        for frame_number in frame_list:
            
            # import pdb;pdb.set_trace()
            if not os.path.isfile(os.path.join(embed_path,area,scene_name, frame_number+'.pth')):
                continue
            if not os.path.isfile(os.path.join(bridge_path,area,scene_name, frame_number+'.npy')):
                continue
            frame_embed = torch.load(os.path.join(embed_path,area,scene_name, frame_number+'.pth'))
            frame_bridge = np.load(os.path.join(bridge_path,area,scene_name, frame_number+'.npy'))
            frame_bridge[:,[0,1]] = frame_bridge[:,[1,0]]
            # import pdb;pdb.set_trace()
            valid_point_list = np.where(frame_bridge[:,2]==1)[0]
            for prompt_number in range(prompt.shape[0]):
                # if prompt[prompt_number, 4] == -1:
                #     continue
                # if frame_bridge[int(prompt[prompt_number, 0]),2] == 0:
                #     continue
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
                # import pdb;pdb.set_trace()
                for valid_point in valid_point_list:
                    if masks[mask_num, int(frame_bridge[valid_point,0])-1, int(frame_bridge[valid_point,1])-1]:
                        # print(sam_label)
                        # print(valid_point)
                        # print(sam_label)
                        if valid_point in mask_dict:
                            if sam_label.item() in mask_dict[valid_point]:
                                mask_dict[valid_point][sam_label.item()] +=1
                            else:
                                mask_dict[valid_point][sam_label.item()] =1
                        else:
                            mask_dict[valid_point] = dict()
                            mask_dict[valid_point][sam_label.item()] =1
                        # import pdb;pdb.set_trace()
                        
                        sam_label_pcd[valid_point,0] = sorted(mask_dict[valid_point].items(), key=lambda x:x[1], reverse=True)[0][0]                    
                        if len(mask_dict[valid_point]) > 1:  # No voting, drop
                            sam_label_pcd[valid_point,0] = -1
        # for frame_number in frame_list2:
        #     if not os.path.isfile(os.path.join(embed_path2,area,scene_name, frame_number+'.pth')):
        #         continue
        #     if not os.path.isfile(os.path.join(bridge_path2,area,scene_name, frame_number+'.npy')):
        #         continue
        #     frame_embed = torch.load(os.path.join(embed_path2,area,scene_name, frame_number+'.pth'))
        #     frame_bridge = np.load(os.path.join(bridge_path2,area,scene_name, frame_number+'.npy'))
        #     frame_bridge[:,[0,1]] = frame_bridge[:,[1,0]]
        #     # import pdb;pdb.set_trace()
        #     valid_point_list = np.where(frame_bridge[:,2]==1)[0]
        #     for prompt_number in range(prompt.shape[0]):
        #         # if prompt[prompt_number, 4] == -1:
        #         #     continue
        #         # if frame_bridge[int(prompt[prompt_number, 0]),2] == 0:
        #         #     continue
        #         if prompt[prompt_number] == 0:
        #             continue
                
        #         if pcd_data['semantic_gt'][prompt_number] == -1:
        #             continue
        #         if frame_bridge[int(prompt_number),2] == 0:
        #             continue
                
        #         sam_predictor.features = frame_embed.cuda()
        #         sam_input_point = np.expand_dims(np.array([frame_bridge[int(prompt_number),1],frame_bridge[int(prompt_number),0]]).astype(int),0)
                
        #         masks, scores, logits = sam_predictor.predict(point_coords=sam_input_point,point_labels=np.array([1]))
        #         sam_label = pcd_data['semantic_gt'][prompt_number]
        #         # import pdb;pdb.set_trace()
        #         for valid_point in valid_point_list:
        #             if masks[mask_num, int(frame_bridge[valid_point,0])-1, int(frame_bridge[valid_point,1])-1]:
        #                 # print(sam_label)
        #                 # print(valid_point)
        #                 # print(sam_label)
        #                 if valid_point in mask_dict:
        #                     if sam_label.item() in mask_dict[valid_point]:
        #                         mask_dict[valid_point][sam_label.item()] +=1
        #                     else:
        #                         mask_dict[valid_point][sam_label.item()] =1
        #                 else:
        #                     mask_dict[valid_point] = dict()
        #                     mask_dict[valid_point][sam_label.item()] =1
        #                 # import pdb;pdb.set_trace()
                        
        #                 sam_label_pcd[valid_point,0] = sorted(mask_dict[valid_point].items(), key=lambda x:x[1], reverse=True)[0][0]                    
        #                 if len(mask_dict[valid_point]) > 1:  # No voting, drop
        #                     sam_label_pcd[valid_point,0] = -1
        color = np.zeros_like(scene_pcd_np)
        for prompt_number in range(prompt.shape[0]):
            if prompt[prompt_number] == 0:
                continue            
            if pcd_data['semantic_gt'][prompt_number] == -1:
                continue
            sam_label_pcd[int(prompt_number),0] = int(pcd_data['semantic_gt'][prompt_number])
            color[int(prompt[prompt_number]),:] = label_to_colors[sam_label_pcd[prompt_number,0]]
        # color = np.zeros_like(scene_pcd_np)
        # write_ply('sam_label_prompt29.ply',[scene_pcd_np, color],['x','y','z','red','green','blue'])
        # color = np.zeros_like(scene_pcd_np)
        # for num in range(sam_label_pcd.shape[0]):
        #     # import pdb;pdb.set_trace()
        #     color[num,:] = label_to_colors[sam_label_pcd[num,0]]
        # write_ply('render_sam_label_bridge2.ply',[scene_pcd_np, color],['x','y','z','red','green','blue'])
        # import pdb;pdb.set_trace()
        print(area ,scene_name)
        np.save(os.path.join(save_path,area, scene_name+'.npy'), sam_label_pcd)
        # import pdb;pdb.set_trace()
