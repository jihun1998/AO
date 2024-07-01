# AO
Official repository for CVPR 2024 paper: "**Weakly Supervised Point Cloud Semantic Segmentation via Artificial Oracle**" by [Hyeokjun Kweon](https://scholar.google.com/citations?user=em3aymgAAAAJ&hl=en&oi=ao) and [Jihun Kim](https://scholar.google.com/citations?user=8UVicysAAAAJ&hl=ko).
This code is implemented based on [pointcept repository](https://github.com/Pointcept/Pointcept/tree/main).

## Prerequisite
* Tested on Ubuntu 20.04, with Python 3.8, PyTorch 1.12.1, CUDA 11.6 with 4 GPUs.
* [S3DIS](https://github.com/alexsax/2D-3D-Semantics) dataset: Download dataset and place under ./data/S2D3D folder.
* [PointTransformer V2](https://github.com/Pointcept/Pointcept/tree/main): Prepare environment and github repository introduced for "PointTransformer V2". Also, you need to follow "Data Preparation" for S3DIS dataset and place the resultant files in ./data/s3dis folder.
* Please install [SAM](https://github.com/facebookresearch/segment-anything) and download vit_h version as ./pretrained/sam_vit_h_4b8939.pth
* You need to run PP2S module using SAM as preprocessing. You need to get "SAM embeddings, bridges for image and point cloud, weak_labels, basket and sam_lables". Please refer to "my_decode_embedding_final.py, my_make_bridge_final.py, my_choose_weak_label_final.py, my_make_basket_final.py, my_run_sam_final.py" in ./pointcept/utils/~.
* You need "used image lists" and "angle and center for alignment". You can download in [used_images](https://drive.google.com/drive/folders/1sjH7DYl5OagmFtIHeTEN62daZXOed0cA?usp=sharing) and [align_angle_and_center](https://drive.google.com/drive/folders/1lEh0N-cYJypq8wlDRJR4uWFIO-pyH-Fe?usp=sharing). Place them under ./used_images and ./data/align_angle_and_center.

## Training
* For training a semantic segmentation model using the labels enhanced by our PP2S module, use the following command:
```
sh scripts/train_pp2s.sh -g 4 -d s3dis -c semseg-pt-v2m2-0-base -n {exp_name}
```
* For training the REAL framework, we initialize it by the semantic segmentation model trained with the PP2S module.
* Hence, you need to place the checkpoint obtained by PP2S under ./pretrained folder. Then, use the following command:
```
sh scripts/train_real.sh -g 4 -d s3dis -c semseg-pt-v2m2-0-sam-final -n {exp_name}
```

## Citation
If our code be useful for you, please consider citing our CVPR 2024 paper using the following BibTeX entry.
```
@inproceedings{kweon2024weakly,
  title={Weakly Supervised Point Cloud Semantic Segmentation via Artificial Oracle},
  author={Kweon, Hyeokjun and Kim, Jihun and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3721--3731},
  year={2024}
}
```
