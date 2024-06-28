"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

@DATASETS.register_module()
class S3DISDataset(Dataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        weak=False,
        weak_path=None,
        mode='pp2s'
    ):
        super(S3DISDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        self.weak = weak
        self.weak_path = weak_path
        self.mode = mode
        print(self.weak_path)

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = sorted(glob.glob(os.path.join(self.data_root, self.split, "*.pth")))
            
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        
        ### Weak label ###
        # SAM_all_levels
        # if self.weak:
        #     temp = data_path.split('/')
        #     area_str = temp[-2]
        #     room_str = temp[-1][:-4]
        #     sam_fly_path = '/home/vilab/khj/ssd0/pointcept/data/sam_labels_on_the_fly/'+area_str+'/'+room_str+'.npy'
        #     vote_all = np.load(sam_fly_path) # (N,4,C)
        #     vote = vote_all[:,0,:] # (N,C)
        #     sam_label = np.argmax(vote, axis=1) # (N,)
        #     sam_label[np.sum(vote,axis=1)==0] = -1

        #     weak_idx = np.load('/home/vilab/khj/ssd0/pointcept/data/weak_labels/'+area_str+'/'+room_str+'.npy')
        #     sam_label[weak_idx==1] = segment[weak_idx==1]
        #     segment = sam_label

        # OTOC
        # if self.weak:
        #     temp = data_path.split('/')
        #     area_str = temp[-2]
        #     room_str = temp[-1][:-4]
        #     weak_label_path = '/home/vilab/khj/ssd0/pointcept/data/weak_labels/'+area_str+'/'+room_str+'.npy'
        #     weak_label = np.load(weak_label_path)
        #     segment[weak_label==0] = -1

        # OTOC_0.02
        # if self.weak:
        #     temp = data_path.split('/')
        #     area_str = temp[-2]
        #     room_str = temp[-1][:-4]
        #     weak_label_path = '/home/vilab/khj/ssd0/pointcept/data/weak_labels_0.02/'+area_str+'/'+room_str+'.npy'
        #     weak_label = np.load(weak_label_path)
        #     segment[weak_label==0] = -1
        
        # SAM
        if self.weak and self.mode == 'pp2s':
            temp = data_path.split('/')
            area_str = temp[-2]
            room_str = temp[-1][:-4]
            sam_label_path = self.weak_path+'/'+area_str+'/'+room_str+'.npy'
            # sam_label_path = '/mnt/jihun4/pointceptHJ/data/sam_labels_mobile'+'/'+area_str+'/'+room_str+'.npy'
            sam_label = np.load(sam_label_path)
            data_dict['segment'] = sam_label.reshape([-1])
            original_idx = np.arange(coord.shape[0])
            data_dict['instance'] = original_idx

        # SAM_0.02
        # if self.weak:
        #     temp = data_path.split('/')
        #     area_str = temp[-2]
        #     room_str = temp[-1][:-4]
        #     sam_label_path = '/home/vilab/khj/ssd0/pointcept/data/sam_labels_0.02'+'/'+area_str+'/'+room_str+'.npy'
        #     sam_label = np.load(sam_label_path)
        #     data_dict['segment'] = sam_label.reshape([-1])

        # SAM RENDER
        # if self.weak:
        #     temp = data_path.split('/')
        #     area_str = temp[-2]
        #     room_str = temp[-1][:-4]
        #     # sam_label_path = '/mnt/jihun4/pointceptHJ/data/rendered_sam_labels2'+'/'+area_str+'/'+room_str+'.npy'
        #     sam_label_path = '/mnt/jihun4/pointceptHJ/exp/s3dis/render_real2_grid.5_th.95/sam_labels_on_the_fly'+'/'+area_str+'/'+room_str+'.npy'
        #     sam_label = np.load(sam_label_path)
        #     data_dict['segment'] = sam_label.reshape([-1])
        #     original_idx = np.arange(coord.shape[0])
        #     data_dict['instance'] = original_idx

        # REAL
        if self.weak and self.mode == 'real':
            temp = data_path.split('/')
            area_str = temp[-2]
            room_str = temp[-1][:-4]
            sam_label_path = self.weak_path+'/'+area_str+'/'+room_str+'.npy'
            sam_label = np.load(sam_label_path)
            data_dict['segment'] = sam_label.reshape([-1])
            original_idx = np.arange(coord.shape[0])
            data_dict['instance'] = original_idx
            
        ##################

        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
