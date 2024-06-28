"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial

import pickle

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage

import pdb
import time
import glob
import numpy as np
from PIL import Image
from pointcept.utils.ply import *
from pointcept.engines.my_evaluate import get_miou
from segment_anything import sam_model_registry, SamPredictor

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
       

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

        # For updating SAM label
        self.prompt_dict_full = {}
        self.seg_pred_dict_full = {}
        
        # Initialize SAM
        self.logger.info("=> Building SAM ...")
        if comm.is_main_process():
            sam_checkpoint = "/home/vilab/khj/ssd0/pointcept/pretrained/sam_vit_h.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to('cuda')
            sam.eval()
            self.predictor = SamPredictor(sam)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict, prompt_dict, seg_pred_dict = self.model(input_dict, get_seg_pred=True)
            self.prompt_dict_full = {**self.prompt_dict_full, **prompt_dict}
            self.seg_pred_dict_full = {**self.seg_pred_dict_full, **seg_pred_dict}
            loss = output_dict["loss"]

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        local_rank = comm.get_local_rank()

        pickle_path = "/home/vilab/khj/ssd0/pointcept/pickle_path/"+str(local_rank)+'_prompt.pickle'
        with open(pickle_path, 'wb') as pp:
            pickle.dump(self.prompt_dict_full, pp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Save prompt dictionary of node ' + str(local_rank) + ' on ' + pickle_path)

        pickle_path = "/home/vilab/khj/ssd0/pointcept/pickle_path/"+str(local_rank)+'_seg_pred.pickle'
        with open(pickle_path, 'wb') as pp:
            pickle.dump(self.seg_pred_dict_full, pp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Save seg_pred dictionary of node ' + str(local_rank) + ' on ' + pickle_path)

        time.sleep(20)
        
        ### SAM label refinement
        

        if comm.is_main_process():
            print("SAM-label refinement start")
            
            # Roots
            data_3d_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis'
            rgb_2d_root = '/home/vilab/khj/ssd0/pointcept/data/S2D3D'
            bridge_root = '/home/vilab/khj/ssd0/pointcept/data/bridge'
            weak_labels_root = '/home/vilab/khj/ssd0/pointcept/data/weak_labels'
            sam_labels_root = '/home/vilab/khj/ssd0/pointcept/data/sam_labels_on_the_fly'
            embedding_root = '/home/vilab/khj/ssd0/pointcept/data/embeddings'
            
            prompt_dict_full = {}
            seg_pred_dict_full = {}
            for ridx in range(4):
                pickle_path = "/home/vilab/khj/ssd0/pointcept/pickle_path/"+str(ridx)+'_prompt.pickle'
                with open(pickle_path, 'rb') as pp:
                    dict_temp = pickle.load(pp)
                prompt_dict_full = {**prompt_dict_full, **dict_temp}

                pickle_path = "/home/vilab/khj/ssd0/pointcept/pickle_path/"+str(ridx)+'_seg_pred.pickle'
                with open(pickle_path, 'rb') as pp:
                    dict_temp = pickle.load(pp)
                seg_pred_dict_full = {**seg_pred_dict_full, **dict_temp}  

            for scene_id, prompts in prompt_dict_full.items():

                pidx_to_cls={}
                for tc, tp in prompts.items():
                    pidx_to_cls[tp] = tc
                
                area_str = scene_id.split('/')[-2]
                room_str = scene_id.split('/')[-1][:-4] # Remove '.pth'
                
                weak_idx = np.asarray(list(prompts.values()))
                print(scene_id, weak_idx)

                if weak_idx.shape[0]!=0:
                    update_flag = False
                    bridge_paths = sorted(glob.glob(bridge_root+'/'+area_str+'/'+room_str+'/*.npy'))

                    # For debug
                    ##
                    # rgb_path_all = rgb_2d_root+'/'+area_str+'/data/rgb'
                    # coord = torch.load('/home/vilab/khj/ssd0/pointcept/data/s3dis/Area_4/hallway_9.pth')['coord']
                    # label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0], [255,255,255]]
                    # palette = np.asarray(label_to_colors)
                    ##
                    
                    for bidx, bridge_path in enumerate(bridge_paths):

                        bridge = np.load(bridge_path)

                        if bidx==0:
                            vote = np.zeros_like(bridge[:,:1]).repeat(13,1) # (N,13)

                        # For debug
                        ##
                        # rgb_path = rgb_path_all + '/' + bridge_path.split('/')[-1][:-4] + '.png'
                        # rgb = np.array(Image.open(rgb_path))
                        ##

                        height = 1080
                        width = 1080

                        viewable_idx = bridge[:,2]==1
                        viewable_coord = bridge[:,:2]

                        viewable_weak_idx = weak_idx[viewable_idx[weak_idx]]

                        # viewable_weak_coord = viewable_coord[viewable_weak_idx] # (V,2)

                        if viewable_weak_idx.shape[0]!=0:
                            update_flag = True
                            viewable_weak_coord = viewable_coord[viewable_weak_idx]

                            emb_path = embedding_root + '/' + area_str + '/' + room_str + '/' + bridge_path.split('/')[-1][:-4] + '.pth'                  
                            emb = torch.load(emb_path) # (1,256,64,64)
                            self.predictor.features = emb.cuda()
                            self.predictor.original_size = (height, width)
                            self.predictor.is_image_set = True
                            self.predictor.input_size = (height, width)

                            input_points_np = viewable_weak_coord.astype(np.float32)
                            input_points_tf = self.predictor.transform.apply_coords(input_points_np, self.predictor.original_size)
                            input_points = torch.from_numpy(input_points_tf).unsqueeze(1).cuda() # (V,1,2)
                            input_labels = torch.ones_like(input_points[:,:,0])
                            
                            masks, _, _ = self.predictor.predict_torch(input_points, input_labels, multimask_output=True)
                            masks = masks.cpu().detach().numpy()
                            
                            for midx, m in enumerate(masks):
                                cls_now = pidx_to_cls[viewable_weak_idx[midx]] # Get from prompt
                                mask_now = m[0]
                                mask_now[0,0] = False
                                sam_expanded_regions = mask_now[viewable_coord[:,1], viewable_coord[:,0]]
                                vote[sam_expanded_regions,cls_now] += 1
                
                    if update_flag:
                        sam_result = np.argmax(vote,axis=1)
                        sam_result[np.sum(vote,axis=1)==0] = -1

                        # Update original label
                        sam_label_path = sam_labels_root+'/'+area_str+'/'+room_str+'.npy'
                        sam_label_ori = np.load(sam_label_path)

                        mask_valid = sam_result != -1
                        print(mask_valid.sum())

                        ## check by model
                        temp = seg_pred_dict_full[scene_id]
                        seg_pred_now = temp[0]
                        original_idx = temp[1]
                        mask_check_by_model = (sam_result[original_idx] == seg_pred_now)
                        mask_valid[original_idx]=mask_check_by_model
                        print(mask_valid.sum())
                        ##

                        sam_label_ori[mask_valid,0] = sam_result[mask_valid]
                        np.save(sam_label_path, sam_label_ori)
               
                # For debug
                ##
                 # write_ply('./temp.ply', [coord, palette[sam_result].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                # pdb.set_trace()
                ##

            mIoU, mPre, mRec, iou_class = get_miou()
            self.writer.add_scalar('sam_label/mIoU', mIoU, self.epoch)
            self.writer.add_scalar('sam_label/mPre', mPre, self.epoch)
            self.writer.add_scalar('sam_label/mRec', mRec, self.epoch)
        else:
            print(comm.get_local_rank())
            time.sleep(20)

        self.prompt_dict_full = {}
        self.seg_pred_dict_full = {}
                
        super().after_epoch()


    def build_model(self):
        model = build_model(self.cfg.model)
        temp = torch.load("/home/vilab/khj/ssd0/pointcept/exp/s3dis/semseg-pt-v2m2-0-sam-level-0/model/model_best.pth")['state_dict']
        sd = {}
        for k,v in temp.items():
            sd[k[7:]] = v.cpu()
        model.load_state_dict(sd)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        self.cfg.data.train.weak = True
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=False),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )

        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler
