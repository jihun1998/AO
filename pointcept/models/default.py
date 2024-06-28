import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from .builder import MODELS, build_model
import pdb

from pointcept.utils.ply import *

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']




@MODELS.register_module()
class DefaultSegmentorSAM_Image(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.count = 0

    def forward(self, input_dict):
                
        seg_logits = self.backbone(input_dict) # (N,C), C=13 in S3DIS
        seg_dict = {}
        
        # train
        if self.training:
            with torch.no_grad():
                scene_id = input_dict['scene_id']
                original_idx = input_dict['instance']
                offset = input_dict['offset']
                seg_gt = input_dict['segment']
                num_offset = offset.shape[0]
                for oidx in range(num_offset):

                    idx_start = 0 if oidx==0 else offset[oidx-1]
                    idx_end = offset[oidx]

                    prompt_dict_now = {}
                    scene_id_now = scene_id[oidx]
                    original_idx_now = original_idx[idx_start:idx_end]
                    seg_gt_now = seg_gt[idx_start:idx_end]
                    cls_gt_now = seg_gt_now.unique()
                    if cls_gt_now[0]==-1:
                        cls_gt_now = cls_gt_now[1:]               
                    
                    seg_logits_now = seg_logits[idx_start:idx_end] # (n,C)

                    seg_dict_key_now = scene_id_now.replace('/','_')[:-4]
                    seg_dict[seg_dict_key_now] = (seg_logits_now, original_idx_now)
                       
            #
            # gt_seg = input_dict['segment']
            # unknown = gt_seg==-1
            # pred_unknown = seg_logits[unknown]
             
            # top2, top2_idx = torch.topk(pred_unknown.softmax(1), 2, dim=1)
            # conf = top2[:,0]-top2[:,1]
            # valid = conf>0.95
            
            # temp_idx = torch.arange(gt_seg.shape[0]).cuda()
            # gt_seg[temp_idx[unknown][valid]] = seg_logits[unknown][valid].argmax(-1)
            #
            loss = self.criteria(seg_logits, input_dict["segment"])

            return dict(loss=loss), seg_dict

        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)







# @MODELS.register_module()
# class DefaultSegmentorSAM_Image(nn.Module):
#     def __init__(self, backbone=None, criteria=None):
#         super().__init__()
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)
#         self.count = 0

#     def forward(self, input_dict, get_seg_pred=False, epoch=0, save_dir=None, rank=0, basket=None):
                
#         seg_logits = self.backbone(input_dict) # (N,C), C=13 in S3DIS
        
#         # train
#         if self.training:
#             with torch.no_grad():
#                 scene_id = input_dict['scene_id']
#                 original_idx = input_dict['instance']
#                 offset = input_dict['offset']
#                 seg_gt = input_dict['segment']
#                 num_offset = offset.shape[0]
#                 for oidx in range(num_offset):

#                     idx_start = 0 if oidx==0 else offset[oidx-1]
#                     idx_end = offset[oidx]

#                     prompt_dict_now = {}
#                     scene_id_now = scene_id[oidx]
#                     original_idx_now = original_idx[idx_start:idx_end]
#                     seg_gt_now = seg_gt[idx_start:idx_end]
#                     cls_gt_now = seg_gt_now.unique()
#                     if cls_gt_now[0]==-1:
#                         cls_gt_now = cls_gt_now[1:]               
                    
#                     seg_logits_now = seg_logits[idx_start:idx_end] # (n,C)
#                     torch.save(seg_logits_now.cpu(), save_dir+"/seg_logit/"+str(rank)+'_'+str(self.count).zfill(4)+'_'+scene_id_now.replace('/','_'))
#                     torch.save(original_idx_now.cpu(), save_dir+"/original_idx/"+str(rank)+'_'+str(self.count).zfill(4)+'_'+scene_id_now.replace('/','_'))
#                     self.count += 1
                        
#             loss = self.criteria(seg_logits, input_dict["segment"])

#             return dict(loss=loss)

#         # eval
#         elif "segment" in input_dict.keys():
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)








@MODELS.register_module()
class DefaultSegmentorSAM(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict, get_seg_pred=False, epoch=0):
                
        seg_logits = self.backbone(input_dict) # (N,C), C=13 in S3DIS
        
        # train
        if self.training:
            if epoch%5==0:
                with torch.no_grad():
                    prompt_dict = {}
                    if get_seg_pred:
                        seg_pred_dict = {}
                    scene_id = input_dict['scene_id']
                    original_idx = input_dict['instance']
                    offset = input_dict['offset']
                    seg_gt = input_dict['segment']
                    num_offset = offset.shape[0]
                    for oidx in range(num_offset):

                        idx_start = 0 if oidx==0 else offset[oidx-1]
                        idx_end = offset[oidx]

                        prompt_dict_now = {}
                        scene_id_now = scene_id[oidx]
                        original_idx_now = original_idx[idx_start:idx_end]
                        seg_gt_now = seg_gt[idx_start:idx_end]
                        cls_gt_now = seg_gt_now.unique()
                        if cls_gt_now[0]==-1:
                            cls_gt_now = cls_gt_now[1:]               
                        
                        seg_logits_now = seg_logits[idx_start:idx_end] # (n,C)
                        
                        # Naive confidence
                        val, ind = torch.topk(seg_logits_now, 2) # each is (n,2) 
                        seg_pred = ind[:,0] # (n,)
                        confidence = val[:,0] - val[:,1] # Naive one
                        
                        # Softmax-based
                        # seg_logits_now_softmax = seg_logits_now.softmax(dim=1)
                        # val, ind = torch.topk(seg_logits_now_softmax, 2) # each is (n,2) 
                        # seg_pred = ind[:,0] # (n,)
                        # confidence = val[:,0] - val[:,1] # Naive one

                        # Entropy-based 
                        # TODO

                        for cidx in cls_gt_now:
                            mask_now = seg_pred==cidx
                            if mask_now.sum()!=0:
                                max_conf_temp, max_ind_temp_mask = torch.max(confidence[mask_now], dim=0)
                                max_ind_temp = torch.nonzero(mask_now)[max_ind_temp_mask]
                                # If weak-label of the current iteration already includes the newly found point, just ignore it.
                                # Threshold value is 1 for now. Tune it.. or adaptive?
                                if max_conf_temp>1 and seg_gt_now[max_ind_temp]!=cidx:
                                    prompt_dict_now[cidx.item()] = original_idx_now[max_ind_temp].item()
                        prompt_dict[scene_id_now] = prompt_dict_now

                        if get_seg_pred:
                            temp = torch.cat((seg_pred.unsqueeze(0),original_idx_now.unsqueeze(0)),dim=0)
                            seg_pred_dict[scene_id_now] = temp.cpu().detach().numpy()


                loss = self.criteria(seg_logits, input_dict["segment"])

                if get_seg_pred:
                    return dict(loss=loss), prompt_dict, seg_pred_dict
                else:
                    return dict(loss=loss), prompt_dict
            
            else:
                loss = self.criteria(seg_logits, input_dict["segment"])
                return dict(loss=loss)

        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)





@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
