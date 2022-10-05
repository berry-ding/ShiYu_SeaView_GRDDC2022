# dataset settings
dataset_type = 'RDDDataset'
data_root = 'datasets/RDD2022/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_scale = [(640, 640), (720,720), (800, 800), (880, 880)]
# img_scale = [(320, 320), (480,480), (560,560), (640, 640), (720,720), (800, 800), (880, 880)]
img_scale = (800, 800)
train_pipeline = [  
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(640, 640),(720, 720),(960,960)],
        # img_scale = [(640, 640), (720,720), (800, 800), (880, 880)],
        img_scale = (640, 640),
        # img_scale = (1280, 1280),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'Czech/train/Czech.txt', data_root + 'India/train/India.txt', data_root + 'Japan/train/Japan.txt',
                data_root + 'Norway/train/Norway.txt', data_root + 'United_States/train/United_States.txt', 
                data_root + 'China_Drone/train/China_Drone.txt',
                data_root + 'China_MotorBike/train/China_MotorBike.txt',
            ],
            img_prefix=[data_root + 'Czech/train/', data_root + 'India/train/', data_root + 'Japan/train/',
                        data_root + 'Norway/train/',  data_root + 'United_States/train/', data_root + 'China_Drone/train/',
                        data_root + 'China_MotorBike/train/',
                        ],
            pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
            ),
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=[data_root + 'Czech/train/val.txt',],
        img_prefix=[data_root + 'Czech/train/',],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=[data_root + 'Czech/train/val.txt',],
        img_prefix=[data_root + 'Czech/train/',],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
