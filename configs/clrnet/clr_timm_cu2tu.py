net = dict(type='Detector', )

backbone=dict(
    type='TIMMBackbone',
    model_name='convnextv2_atto.fcmae_ft_in1k',
    features_only=True,
    pretrained=False,
    use_hack_weight=False,
    out_indices=[0, 1, 2, 3],
    hack_weight_path="/home/fvf6zk6/Workspace/Pretrained-Weights/timm/convnextv2_atto.fcmae_ft_in1k_converted.pth",
)

num_points = 72
max_lanes = 5
sample_y = range(710, 150, -1)

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
ema = False


iou_loss_weight = 3.0
cls_loss_weight = 0.1
xyt_loss_weight = 0.1
seg_loss_weight = 0.2
reg_loss_weight = [12.0, 6.0, 3.0]
num_branch_loss_weight = 1.0
num_lane_loss_weight = 1.0
tri_loss_weight = 2.0 # fixed
seg_dist_weight = 0.003

work_dirs = "work_dirs/clr/r18_tusimple"

neck = dict(type='FPN',
            in_channels=[80, 160, 320],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.2, nms_thres=50, nms_topk=max_lanes)
pseudo_label_parameters = dict(conf_threshold=0.5, nms_thres=50, max_lanes=10, nlane=5)
rectify_parameters = dict(upper_thr = 0.5, lower_thr = 0.1)

epochs = 1
batch_size = 40

optimizer = dict(type='AdamW', lr=1.0e-5)  # 3e-4 for batchsize 8
total_iter = (3268 * 3 // batch_size + 1) * epochs
scheduler = dict(type = 'CosineAnnealingLR', T_max = total_iter)

eval_from = 0 # must smaller than epochs
eval_ep = 1

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
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

dataset_path = './data/tusimple'
dataset_type = 'TuSimple'
test_json_file = 'data/tusimple/test_label.json'
source_dataset_path = './data/CULane'
source_dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
    teacher_process=val_process,
    repeat_factor=3,
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
source=dict(
    type=source_dataset_type,
    data_root=source_dataset_path,
    split='train',
    processes=train_process,
    data_size=3268 * 3,
),
)

workers = 10
log_interval = 1
# seed = 0
num_classes = 4
seg_weight = [0.4, 1.0, 1.0, 1.0]
ignore_label = 255
# bg_weight = 0.4
lr_update_by_epoch = False