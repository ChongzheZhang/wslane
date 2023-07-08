net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)
featuremap_out_channel = 512 
featuremap_out_stride = 32 

num_points = 72
max_lanes = 13
sample_y = range(589, 230, -1)

heads = dict(type='LaneATT',
             anchors_freq_path=None,)

seg_branch = True

train_parameters = dict(
    conf_threshold=None,
    nms_thres=15.,
    nms_topk=3000
)
test_parameters = dict(
    conf_threshold=0.4,
    nms_thres=15,
    nms_topk=max_lanes
)

optimizer = dict(
  type = 'AdamW',
  lr = 0.0015,
)

epochs = 2
batch_size = 40
total_iter = (10000 // batch_size) * epochs
scheduler = dict(type = 'CosineAnnealingLR', T_max = total_iter)

eval_from = epochs - 15 # must smaller than epochs
eval_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
img_w=640
img_h=360
cut_height=0

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name = 'HorizontalFlip', parameters = dict(p=1.0), p=0.5),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
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
    dict(type='ToTensor', keys=['img', 'lane_line']),
]

dataset_path = './data/Curvelanes'
dataset_type = 'Curvelane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
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
    debug=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='debug',
        processes=train_process,
    ),
)


workers = 12
log_interval = 1
seed=0
lr_update_by_epoch = False
num_classes = 4
seg_weight = [0.5, 1.0, 1.0, 1.5]
