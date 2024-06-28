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
import pickle

CATS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

label_to_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],[0,0,0], [255,255,255]]
palette = np.asarray(label_to_colors)

data_3d_root = '../../data/s3dis'
area_3d_paths = sorted(glob.glob(data_3d_root+'/*'))

basket = {}
for area_3d_path in area_3d_paths:
    area_str = area_3d_path.split('/')[-1]

    if area_str!='Area_5':

        room_paths = sorted(glob.glob(area_3d_path+'/*.pth'))
        
        for room_path in room_paths:

            room_str = room_path.split('/')[-1][:-4]

            basket_key = 'data_s3dis_'+area_str+'_'+room_str

            data_3d = torch.load(room_path)
            coord = data_3d['coord']
            basket[basket_key] = -100*np.ones((coord.shape[0],13))

pickle_path = '../../data/basket_s3dis.pickle'
with open(pickle_path, 'wb') as pp:
    pickle.dump(basket, pp, protocol=pickle.HIGHEST_PROTOCOL)