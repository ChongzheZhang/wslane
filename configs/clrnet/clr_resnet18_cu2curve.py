net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 13
sample_y = range(589, 230, -1)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

ws_learn = True
ws_combine_learn = True
num_branch = False
seg_branch = True
tri_loss = True
kd_learn = True
ema = True

iou_loss_weight = 3.0
cls_loss_weight = 0.1
xyt_loss_weight = 0.1
seg_loss_weight = 1.0
reg_loss_weight = [12.0, 6.0, 3.0]
num_branch_loss_weight = 1.0
num_lane_loss_weight = 1.0
tri_loss_weight = 1.0


neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.8, nms_thres=15, nms_topk=max_lanes)
pseudo_label_parameters = dict(conf_threshold=0.5, nms_thres=15, max_lanes=10, nlane=max_lanes)
rectify_parameters = dict(upper_thr = 0.5, lower_thr = 0.1)

epochs = 1
batch_size = 40

optimizer = dict(type='AdamW', lr=4.0e-5)  # 3e-4 for batchsize 8
total_iter = (100000 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_from = 10
eval_ep = 10

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 160

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/Curvelanes'
dataset_type = 'Curvelane'
source_dataset_path = './data/CULane'
source_dataset_type = 'CULane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
        teacher_process=val_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='valid',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='valid',
        processes=val_process,
    ),
    source=dict(
        type=source_dataset_type,
        data_root=source_dataset_path,
        split='train',
        processes=train_process,
        data_size=100000,
        repeat_factor=2,
    )
)

workers = 10
log_interval = 1
# seed = 0
num_classes = 4
seg_weight = [0.4, 1.0, 1.0, 1.0]
ignore_label = 255
lr_update_by_epoch = False
