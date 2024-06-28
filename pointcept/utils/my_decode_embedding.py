import os
import glob
import torch
import numpy as np
import pickle

emb_encoded_root = '/home/vilab/khj/ssd0/pointcept/data/embeddings_encoded'
emb_root = '/home/vilab/khj/ssd0/pointcept/data/embeddings'
emb_encoded_paths = sorted(glob.glob(emb_encoded_root+'/*.pkl'))

for emb_encoded_path in emb_encoded_paths:
    area_str = emb_encoded_path.split('/')[-1][:6]
    print(area_str)
    os.makedirs(emb_root+'/'+area_str, exist_ok=True)

    with open(emb_encoded_path, 'rb') as eep:
        emb_encoded = pickle.load(eep)[0]
        
    for room_str, emb_dict in emb_encoded.items():
        print(room_str)
        os.makedirs(emb_root+'/'+area_str+'/'+room_str, exist_ok=True)

        for rgb_str, emb in emb_dict.items():
            save_path = emb_root+'/'+area_str+'/'+room_str+'/'+rgb_str+'.pth'
            torch.save(emb, save_path)
