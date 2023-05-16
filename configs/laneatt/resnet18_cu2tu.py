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
max_lanes = 5
sample_y = range(710, 150, -10)

heads = dict(type='LaneATT',
             anchors_freq_path=None,
             topk_anchors=1000)

num_branch = True
seg_branch = True
ws_learn = True
tri_loss = True
det_to_seg = True

train_parameters = dict(
    conf_threshold=None,
    nms_thres=15.,
    nms_topk=3000
)
test_parameters = dict(
    conf_threshold=0.3,
    nms_thres=45,
    nms_topk=max_lanes
)
pseudo_label_parameters = dict(
    conf_threshold=0.5,
    nms_thres=45,
    max_lanes=10,
    nlane=5,
)

optimizer = dict(
  type = 'Adam',
  lr = 0.0001,
)

epochs = 1
batch_size = 40
total_iter = (3268 // batch_size + 1) * epochs
scheduler = dict(type='StepLR', step_size=10, gamma=0.99)

eval_from = epochs - 1 # must smaller than epochs
eval_ep = 1

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w=1280
ori_img_h=720
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

dataset_path = './data/tusimple'
test_json_file = 'data/tusimple/test_label.json'
dataset_type = 'TuSimple'
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
        split='val',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    debug=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='debug',
        processes=val_process,
    ),
)


workers = 12
log_interval = 1
seed=0
lr_update_by_epoch = False
num_classes = 4