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

import tqdm
import pdb
import time
import glob
import math
import numpy as np
from PIL import Image
from pointcept.utils.ply import *
from pointcept.engines.my_evaluate import get_miou
from pointcept.utils.misc import intersection_and_union
from segment_anything import sam_model_registry, SamPredictor
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import softmax

ply_field = ['x','y','z','red','green','blue']
CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0], [255,255,255]]
palette = np.asarray(label_to_colors)

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
        
        # Initialize SAM
        self.logger.info("=> Building SAM ...")
        if comm.is_main_process():
            sam_checkpoint = "/home/vilab/khj/ssd0/pointcept/pretrained/sam_vit_h.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to('cuda')
            sam.eval()
            self.predictor = SamPredictor(sam)

        self.log_dir = '/home/vilab/khj/ssd0/pointcept/'+self.cfg.save_path
        
        if comm.is_main_process():
            os.makedirs(self.log_dir+"/seg_logit", exist_ok=True)
            os.makedirs(self.log_dir+"/seg_logit_aggregated", exist_ok=True)
            os.makedirs(self.log_dir+"/original_idx", exist_ok=True)
            print('Making pass done')

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
            output_dict = self.model(input_dict, epoch=self.epoch, save_dir=self.log_dir, rank=comm.get_local_rank())
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

        super().after_epoch()

        self.model.count = 0
        height = 1080
        width = 1080

        ### SAM label refinement       
        if comm.is_main_process():
            print("SAM-label refinement start")
            
            # Roots
            data_3d_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis'
            cls_gt_root = '/home/vilab/khj/ssd0/pointcept/data/s3dis_cls_gt'
            rgb_2d_root = '/home/vilab/khj/ssd0/pointcept/data/S2D3D'
            bridge_root = '/home/vilab/khj/ssd0/pointcept/data/bridge'
            
            embedding_root = '/home/vilab/khj/ssd0/pointcept/data/embeddings'
            sam_labels_root = self.log_dir+'/sam_labels_on_the_fly'
            seg_logit_root = self.log_dir+"/seg_logit"
            seg_logit_agg_root = self.log_dir+"/seg_logit_aggregated"
            original_idx_root = self.log_dir+"/original_idx"

            seg_logit_paths = sorted(glob.glob(seg_logit_root+'/*.pth'))
            count_updated = 0
            prompt_accuracy_all = 0
            
            print("Aggregating predicted seg_logits...")
            aggregated_seg_logit_paths = []
            for ridx, seg_logit_path in enumerate(tqdm.tqdm(seg_logit_paths)):
                temp = seg_logit_path.split('/')[-1].split('_')
                if temp[0]=='Area':
                    continue
                # area_str = temp[2]+'_'+temp[3]
                # room_str = temp[4]+'_'+temp[5][:-4]
                area_str = temp[4]+'_'+temp[5]
                room_str = temp[6]+'_'+temp[7][:-4]

                temp_save_path = seg_logit_agg_root+'/'+area_str+'_'+room_str+'.npy'

                if temp_save_path in aggregated_seg_logit_paths:
                    aggregated_seg_logit = np.load(temp_save_path)
                    seg_logit = torch.load(seg_logit_path).float().numpy()
                    sampled_idx = torch.load(original_idx_root+'/'+seg_logit_path.split('/')[-1]).numpy() # (n,)
                    aggregated_seg_logit[sampled_idx] = seg_logit
                    np.save(temp_save_path, aggregated_seg_logit)

                else:
                    aggregated_seg_logit_paths.append(temp_save_path)
                    pc = torch.load(data_3d_root + '/' + area_str+'/'+room_str+'.pth')
                    coord = pc['coord']
                    aggregated_seg_logit = -100*np.ones_like(coord[:,:1]).repeat(13,1)
                    seg_logit = torch.load(seg_logit_path).float().numpy()
                    sampled_idx = torch.load(original_idx_root+'/'+seg_logit_path.split('/')[-1]).numpy() # (n,)
                    aggregated_seg_logit[sampled_idx] = seg_logit
                    np.save(temp_save_path, aggregated_seg_logit)
            
            # aggregated_seg_logit_paths = glob.glob(seg_logit_agg_root+'/*.npy')
            aggregated_seg_logit_paths = sorted(aggregated_seg_logit_paths)

            print("Done.")

            print("Sampling prompts and enhancing by SAM...")
            for ridx, seg_logit_path in enumerate(tqdm.tqdm(aggregated_seg_logit_paths)):

                flag_updated = False
                
                # Paths
                temp = seg_logit_path.split('/')[-1].split('_')
                area_str = temp[0]+'_'+temp[1]
                room_str = temp[2]+'_'+temp[3][:-4]

                rgb_path_all = rgb_2d_root + '/' + area_str + '/data/rgb'
                bridge_paths = sorted(glob.glob(bridge_root+'/'+area_str+'/'+room_str+'/*.npy'))
                sam_label_path = sam_labels_root+'/'+area_str+'/'+room_str+'.npy'
                sam_label_ori = np.load(sam_label_path)

                # Get seg_logit and compute confidence/prediction
                seg_logit = np.load(seg_logit_path) # (n,13)
                seg_pred = np.argmax(seg_logit, axis=1)
                seg_pred[seg_logit[:,0]==-100] = -1

                seg_logit = softmax(seg_logit, axis=1)
                top_two = np.sort(seg_logit, axis=1)[:,-2:]
                confidence = top_two[:,1] - top_two[:,0]                

                # Get input point cloud
                pc = torch.load(data_3d_root + '/' + area_str+'/'+room_str+'.pth')
                coord = pc['coord']
                gt_semseg = pc['semantic_gt'][:,0]
                cls_gt_now = np.unique(gt_semseg)

                # Define voting array for refining SAM label                
                vote = np.zeros_like(sam_label_ori).repeat(13,1) # N,C

                ##########################################################################################
                ############################### Grid-like prompt searching ###############################
                ##########################################################################################

                max_x, max_y, _ = np.max(coord, axis=0)
                min_x, min_y, _ = np.min(coord, axis=0)

                length_x = max_x - min_x
                length_y = max_y - min_y

                prompt_cls = []
                prompt_idx = []
                grid_scale = 1
                for xidx in range(math.ceil(length_x)//grid_scale):
                    start_x = min_x + (xidx)*grid_scale
                    end_x = min_x + (xidx+1)*grid_scale
                    mask_x = (coord[:,0]>start_x)*(coord[:,0]<end_x)
                    for yidx in range(math.ceil(length_y//grid_scale)):
                        start_y = min_y + (yidx)*grid_scale
                        end_y = min_y + (yidx+1)*grid_scale
                        mask_y = (coord[:,1]>start_y)*(coord[:,1]<end_y)
                        mask_grid = mask_x*mask_y

                        seg_pred_grid = seg_pred[mask_grid]
                        confidence_grid = confidence[mask_grid]
                        sam_label_ori_grid = sam_label_ori[mask_grid]

                        for cidx in cls_gt_now:
                            cls_mask_grid = seg_pred_grid==cidx
                            if cls_mask_grid.sum()>0:
                                incognita = (sam_label_ori_grid[cls_mask_grid,0]!=cidx)
                                if incognita.sum()>0:
                                    confidence_now = confidence_grid[cls_mask_grid][incognita]
                                    max_conf_idx = np.argmax(confidence_now)
                                    
                                    # TODO-for-JHK
                                    if confidence_now[max_conf_idx]>0.95:
                                        prompt_cls.append(cidx)
                                        temp_idx = np.arange(sam_label_ori.shape[0])
                                        prompt_idx.append(temp_idx[mask_grid][cls_mask_grid][incognita][max_conf_idx])

                prompt_cls = np.asarray(prompt_cls)
                prompt_idx = np.asarray(prompt_idx)

                ##########################################################################################
                ################################## SAM-label refinement ##################################
                ##########################################################################################

                if prompt_idx.shape[0]>0:
                    
                    prompt_accuracy = (gt_semseg[prompt_idx]==prompt_cls).sum()/prompt_idx.shape[0]
                    prompt_accuracy_all += prompt_accuracy

                    for bidx, bridge_path in enumerate(bridge_paths):

                        rgb_path = rgb_path_all + '/' + bridge_path.split('/')[-1][:-4] + '.png'
                        flag_vis = ridx%20==0 and (bidx==0 or bidx==10)
                        # rgb = np.array(Image.open(rgb_path))

                        bridge = np.load(bridge_path)
                        idx_viewable = bridge[:,2]==1 # (V,)
                        coord_viewable = bridge[:,:2] # (V,2)

                        gt_semseg_viewable = pc['semantic_gt'][idx_viewable][:,0]
                        seg_pred_viewable = seg_pred[idx_viewable]
                        confidence_viewable = confidence[idx_viewable]

                        prompt_viewable = idx_viewable[prompt_idx]
                        if prompt_viewable.sum()>0:
                            viewable_prompt_coord = coord_viewable[prompt_idx][prompt_viewable]
                            viewable_prompt_cls = prompt_cls[prompt_viewable]

                            # if flag_vis:
                            #     for sidx1 in range(5):
                            #         for sidx2 in range(5):
                            #             rgb[viewable_prompt_coord[:,1]+sidx1-2, viewable_prompt_coord[:,0]+sidx2-2,:] = palette[viewable_prompt_cls]
                            #     self.writer.add_image(area_str+'/'+room_str+'/'+str(bidx).zfill(2), rgb, self.epoch, dataformats='HWC')

                            # print(area_str, room_str, bridge_path, viewable_prompt_cls)

                            flag_updated = True

                            emb_path = embedding_root + '/' + area_str + '/' + room_str + '/' + bridge_path.split('/')[-1][:-4] + '.pth'                  
                            emb = torch.load(emb_path) # (1,256,64,64)
                            self.predictor.features = emb.cuda()
                            self.predictor.original_size = (height, width)
                            self.predictor.is_image_set = True
                            self.predictor.input_size = (height, width)

                            input_points_np = viewable_prompt_coord.astype(np.float32)
                            input_points_tf = self.predictor.transform.apply_coords(input_points_np, self.predictor.original_size)
                            input_points = torch.from_numpy(input_points_tf).unsqueeze(1).cuda() # (V,1,2)
                            input_labels = torch.ones_like(input_points[:,:,0])
                                
                            masks, _, _ = self.predictor.predict_torch(input_points, input_labels, multimask_output=True)
                            masks = masks.cpu().detach().numpy()

                            for midx, m in enumerate(masks):
                                cls_mask = viewable_prompt_cls[midx]
                                mask_now = m[0]
                                mask_now[0,0] = False
                                sam_enhanced_idx = mask_now[coord_viewable[idx_viewable][:,1], coord_viewable[idx_viewable][:,0]]

                                seg_pred_mask = seg_pred_viewable[sam_enhanced_idx]
                                conf_mask = confidence_viewable[sam_enhanced_idx]

                                if (conf_mask>0.9).sum()>0:
                                    cls_pred_mask = stats.mode(seg_pred_mask[conf_mask>0.9],keepdims=False)[0]
                                    # TODO-for-JHK
                                    if cls_mask==cls_pred_mask:
                                        temp = np.arange(vote.shape[0])
                                        vote[temp[idx_viewable][sam_enhanced_idx],cls_mask] += 1
                
                if flag_updated:
                    
                    # intersection, union, gt_true, positive = intersection_and_union(sam_label_ori[:,0], gt_semseg, 13, get_output=True)
                    # print('ori')
                    # iou_ori = intersection/(union+1e-5)
                    # pre_ori = intersection/(positive+1e-5)
                    # rec_ori = intersection/(gt_true+1e-5)
                    
                    # print(intersection/(union+1e-5))
                    # print(intersection/(positive+1e-5))
                    # print(intersection/(gt_true+1e-5))

                    if room_str=='conferenceRoom_1':
                        pc = torch.load(data_3d_root + '/' + area_str+'/'+room_str+'.pth')
                        coord = pc['coord']
                        write_ply(self.log_dir+'/vis_'+area_str+'_'+str(self.epoch).zfill(2)+'_al.ply', [coord, palette[sam_result].astype(np.uint8)], ['x','y','z','red','green','blue'])
                        write_ply(self.log_dir+'/vis_'+area_str+'_'+str(self.epoch).zfill(2)+'_pred.ply', [coord, palette[seg_pred].astype(np.uint8)], ['x','y','z','red','green','blue'])

                    sam_result = np.argmax(vote,axis=1)
                    sam_result[np.sum(vote,axis=1)==0] = -1

                    mask_check_by_model = (sam_result!=seg_pred) + (seg_pred==-1)           
                    sam_result[mask_check_by_model] = -1                   

                    # intersection, union, gt_true, positive = intersection_and_union(sam_result, gt_semseg, 13, get_output=True)
                    # print('this step')
                    # print(intersection/(union+1e-5))
                    # print(intersection/(positive+1e-5))
                    # print(intersection/(gt_true+1e-5))

                    # write_ply('./temp_old.ply', [coord, palette[sam_label_ori[:,0]].astype(np.uint8)], ply_field)

                    mask_valid = sam_result!=-1
                    count_updated += (sam_label_ori[mask_valid,0] != sam_result[mask_valid]).sum()
                    sam_label_ori[mask_valid,0] = sam_result[mask_valid]

                    # write_ply('./temp_new.ply', [coord, palette[sam_label_ori[:,0]].astype(np.uint8)], ply_field)

                    # intersection, union, gt_true, positive = intersection_and_union(sam_label_ori[:,0], gt_semseg, 13, get_output=True)
                    # print('updated')
                    # iou_updated = intersection/(union+1e-5)
                    # pre_updated = intersection/(positive+1e-5)
                    # rec_updated = intersection/(gt_true+1e-5)
                    # print(intersection/(union+1e-5))
                    # print(intersection/(positive+1e-5))
                    # print(intersection/(gt_true+1e-5))  

                    # print('improvement')
                    # print(np.round(iou_updated-iou_ori,3).mean())
                    # print(np.round(pre_updated-pre_ori,3).mean())
                    # print(np.round(rec_updated-rec_ori,3).mean())
                    # pdb.set_trace()

                    np.save(sam_label_path, sam_label_ori)

                if room_str=='conferenceRoom_1':
                    pc = torch.load(data_3d_root + '/' + area_str+'/'+room_str+'.pth')
                    coord = pc['coord']
                    write_ply(self.log_dir+'/vis_'+area_str+'_'+str(self.epoch).zfill(2)+'.ply', [coord, palette[sam_label_ori[:,0]].astype(np.uint8)], ['x','y','z','red','green','blue'])


            if self.epoch==0:
                self.writer.add_scalar('sam_label/mIoU', 0.30396531012703654, -1)
                self.writer.add_scalar('sam_label/mPre', 0.8307590457060119, -1)
                self.writer.add_scalar('sam_label/mRec', 0.3281302311798235, -1)

            mIoU, mPre, mRec, iou_class = get_miou(self.log_dir+'/sam_labels_on_the_fly')
            self.writer.add_scalar('sam_label/mIoU', mIoU, self.epoch)
            self.writer.add_scalar('sam_label/mPre', mPre, self.epoch)
            self.writer.add_scalar('sam_label/mRec', mRec, self.epoch)
            self.writer.add_scalar('sam_label/num_updated', count_updated/204, self.epoch)
            self.writer.add_scalar('sam_label/prompt_accuracy', prompt_accuracy_all/204, self.epoch)

            os.system('rm -r ' + seg_logit_root+'/*')

        else:
            print(comm.get_local_rank())
            time.sleep(100)



    def build_model(self):
        model = build_model(self.cfg.model)
        
        #
        temp = torch.load("/home/vilab/khj/ssd0/pointcept/exp/s3dis/semseg-pt-v2m2-0-sam-level-0/model/model_best.pth")['state_dict']
        sd = {}
        for k,v in temp.items():
            sd[k[7:]] = v.cpu()
        model.load_state_dict(sd)
        #

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
        self.cfg.data.train.weak_path = '/home/vilab/khj/ssd0/pointcept/'+self.cfg.save_path+'/sam_labels_on_the_fly'
        if comm.is_main_process():
            os.system("cp -r /home/vilab/khj/ssd0/pointcept/data/sam_labels " + self.cfg.data.train.weak_path)
            print('on-the-fly weak label path is copied')
        
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
