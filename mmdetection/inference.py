#coding=utf-8
 
from re import L
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
import numpy as np
from mmcv import Config, DictAction
import glob
import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description='...')
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('source', help='test images path')
parser.add_argument('imgsz', help='test imgsz')
parser.add_argument('conf_thres', help='test imgsz')
parser.add_argument('iou_thres', help='test imgsz')
parser.add_argument('filename', help='outputfilename')
args = parser.parse_args()


config_file = args.config
checkpoint_file = args.checkpoint
path = args.source
imgsz = args.imgsz
conf_thres = args.conf_thres
iou_thres = args.iou_thres
filename = args.filename

cfg = Config.fromfile(config_file)
cfg['test_pipeline'][1]['img_scale'] = (imgsz, imgsz)
cfg.model.test_cfg.rcnn.score_thr = float(conf_thres)
cfg.model.test_cfg.rcnn.nms.iou_threshold = float(iou_thres)

model = init_detector(cfg, checkpoint_file, device='cuda:0')
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
    with open('results_mmdet/' + filename, 'a') as w:
        w.write(image_name + ',' + bbox_str)