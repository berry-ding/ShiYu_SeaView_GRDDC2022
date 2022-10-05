# dataset settings
dataset_type = 'RDDDataset'
data_root = 'datasets/RDD2022/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = [(640, 640), (720,720), (800, 800)]
train_pipeline = [  
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(640, 640),(720, 720),(960,960)],
        img_scale = (800, 800),
        flip=False,
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
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
            pipeline=train_pipeline)),
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
