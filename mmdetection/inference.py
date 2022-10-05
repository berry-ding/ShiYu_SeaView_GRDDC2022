#coding=utf-8
 
from re import L
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
# from mmdet.apis import show_result
import numpy as np



# config_file = 'configs/swin/cascade_rcnn_swin.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin/epoch_22.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img = 'datasets/road2020/test1/test_images/Czech_002798.jpg'
# result = inference_detector(model, img)
# # import pdb;pdb.set_trace()
# # import pdb;pdb.set_trace()
# model.show_result(img, result, out_file='test_640flip_Czech_002798.jpg', score_thr=0.1)
# 1599
#test00 C2798

import glob
import tqdm
import os
# config_file = 'configs/swin/faster_swin_l.py'
# checkpoint_file = 'swin_epoch_36.pth'
config_file = 'configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco.py'
checkpoint_file = 'dyhead_epoch_36.pth'
# config_file = 'configs/swin/faster_swin_l12_deform.py'
# checkpoint_file = 'swin12w_deform_epoch_36.pth'
# config_file = 'configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
# checkpoint_file = 'detectors_epoch_20.pth'
path = 'datasets/RDD2022/test1_images/'
# path = 'datasets/RDD2022/Japan/train/images/'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
for image_name in tqdm.tqdm(os.listdir(path)):
    result = inference_detector(model, path+image_name)
    bbox_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bbox_str = ""
    for bbox, label in zip(bboxes, labels):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        label = int(label) + 1
        # if bbox[4] > 0.2:
        bbox_str += str(label) + ' ' + str(xmin) + ' ' \
                    + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + ' '
    bbox_str += '\n'
    # with open('results_mmdet/detectors_22_1_e20_88_iou0.9999_conf0.50.txt', 'a') as w:
    with open('results_mmdet/dyhead_e36_all_72_conf0.05_iou0.6.txt', 'a') as w:
        w.write(image_name + ',' + bbox_str)