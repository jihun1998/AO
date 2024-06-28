import os
import numpy as np
import torch
import sys
sys.path.append('/mnt/jihun4/pointceptHJ')

from pointcept.utils.misc import intersection_and_union

import glob
import tqdm
import pdb
from scipy.special import softmax

gt_root = '/mnt/jihun4/pointceptHJ/data/s3dis'
area_paths = sorted(glob.glob(gt_root+'/*'))
area_paths = ['/mnt/jihun4/pointceptHJ/data/s3dis/Area_1']
def get_miou(pred_root=None):
    count = 0
    for area_path in area_paths:
        area_str = area_path.split('/')[-1]
        if area_str=="Area_5":
            continue
        room_paths = sorted(glob.glob(area_path+'/*.pth'))

        for room_path in tqdm.tqdm(room_paths):
            room_str = room_path.split('/')[-1][:-4]
            pred_path = pred_root+'/'+area_str+'/'+room_str+'.npy'

            if not os.path.isfile(pred_path):
                print(pred_path+' does not exist')
                continue
            
            gt = torch.load(room_path)['semantic_gt']
            pred = np.load(pred_path)

            intersection, union, gt_true, positive = intersection_and_union(pred, gt, 13, get_output=True)

            if count==0:
                inter_all = intersection
                union_all = union
                gt_true_all = gt_true
                positive_all = positive
                count += 1 
            else:
                inter_all += intersection
                union_all += union
                gt_true_all += gt_true
                positive_all += positive

    iou_class = inter_all / (union_all + 1e-10)
    precision_class = inter_all / (positive_all + 1e-10)
    recall_class = inter_all / (gt_true_all + 1e-10)

    print(iou_class)

    mIoU = np.mean(iou_class)
    mPre = np.mean(precision_class)
    mRec = np.mean(recall_class)

    print(mIoU)
    print(mPre)
    print(mRec)

    return mIoU, mPre, mRec, iou_class


def get_miou_from_logit(pred_root=None):
    count = 0
    for area_path in area_paths:
        area_str = area_path.split('/')[-1]
        if area_str=="Area_5":
            continue
        room_paths = sorted(glob.glob(area_path+'/*.pth'))

        for room_path in tqdm.tqdm(room_paths):
            room_str = room_path.split('/')[-1][:-4]
            pred_path = pred_root+'/'+area_str+'_'+room_str+'.npy'

            if not os.path.isfile(pred_path):
                print(pred_path+' does not exist')
                continue
            
            gt = torch.load(room_path)['semantic_gt']
            logit = np.load(pred_path)
            maxs = np.sort(softmax(logit,axis=1), axis=1)[:,-2:]
            conf = maxs[:,1]-maxs[:,0]

            pred = np.argmax(logit, axis=1)
            pred[logit[:,0]==-1]=-1
            pred[conf<0.95] = -1
            pred = np.expand_dims(pred, axis=1)

            intersection, union, gt_true, positive = intersection_and_union(pred, gt, 13, get_output=True)

            if count==0:
                inter_all = intersection
                union_all = union
                gt_true_all = gt_true
                positive_all = positive
                count += 1 
            else:
                inter_all += intersection
                union_all += union
                gt_true_all += gt_true
                positive_all += positive

    iou_class = inter_all / (union_all + 1e-10)
    precision_class = inter_all / (positive_all + 1e-10)
    recall_class = inter_all / (gt_true_all + 1e-10)

    print(iou_class)

    mIoU = np.mean(iou_class)
    mPre = np.mean(precision_class)
    mRec = np.mean(recall_class)

    print(mIoU)
    print(mPre)
    print(mRec)

    pdb.set_trace()

    return mIoU, mPre, mRec, iou_class

if __name__ == "__main__":

    # pred_root = '/home/vilab/khj/ssd0/pointcept/exp/s3dis/please_th0.9/sam_labels_on_the_fly'
    pred_root = '/mnt/jihun4/pointceptHJ/data/sam_labels_mobile'
    get_miou(pred_root)