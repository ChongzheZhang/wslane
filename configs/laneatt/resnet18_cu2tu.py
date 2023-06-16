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

ws_learn = True
ws_combine_learn = False
num_branch = False
seg_branch = True
tri_loss = True
pycda = False
det_to_seg = False
seg_proportion_to_num = False

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
rectify_parameters = dict(
    upper_thr = 0.8,
    lower_thr = 0.3,
)

optimizer = dict(
  type = 'Adam',
  lr = 1e-4,
)

epochs = 1
batch_size = 40
total_iter = (3268 // batch_size + 1) * epochs
scheduler = dict(type='StepLR', step_size=10, gamma=0.99)
# scheduler = dict(type = 'CosineAnnealingLR', T_max = total_iter)

eval_from = 0 # must smaller than epochs
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
source_dataset_path = './data/CULane'
source_dataset_type = 'CULane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
        # repeat_factor=3,
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
    # source=dict(
    #     type=source_dataset_type,
    #     data_root=source_dataset_path,
    #     split='train',
    #     processes=train_process,
    #     data_size=3268 * 3,
    # )
)


workers = 12
log_interval = 1
seed=0
lr_update_by_epoch = False
num_classes = 4
seg_weight = [0.5, 1.0, 1.0, 1.5]

cls_loss_weight = 1.0
reg_loss_weight = 1.0
num_lane_loss_weight = 1.0
seg_loss_weight = 1.0
tri_loss_weight = 1.5
seg_prop_weight = 0.1