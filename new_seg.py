import pickle as pkl
import json
import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

# for culane
cache_path = 'cache/culane_train.pkl'
with open(cache_path, 'rb') as cache_file:
    data_infos = pkl.load(cache_file)

for k, data in enumerate(tqdm(data_infos)):
    mask_path = data['mask_path']
    mask_path = osp.realpath(mask_path)
    path_split = osp.split(mask_path)
    new_path = osp.join(path_split[0], 'bi_mask', path_split[1])
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask[mask>0] = 1
    output = mask * 3
    id = [2, 1]

    for i in id:
        kernel = np.ones((21, 21), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        diff = (dilation - mask) * i
        output += diff
        mask = dilation

    if not osp.exists(osp.dirname(new_path)):
        os.makedirs(osp.dirname(new_path))
    cv2.imwrite(new_path, output)


"""
# for tusimple
ALL_SET = ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json', 'test_label.json']
data_infos = []
for set in ALL_SET:
    set_path = osp.join('./data/tusimple', set)
    with open(set_path, 'r') as f:
        for line in f:
            label = json.loads(line)
            img_path = label['raw_file']
            seg_path = img_path.split('/')
            seg_path = osp.join(os.getcwd(), 'data/tusimple/seg_label', seg_path[1], seg_path[2], seg_path[3])
            seg_path_split = osp.splitext(seg_path)
            right_path = seg_path_split[0] + '.png'
            data_infos.append(right_path)


for k, data in enumerate(tqdm(data_infos)):
    mask_path = data
    mask_path = osp.realpath(mask_path)
    path_split = osp.split(mask_path)
    new_path = osp.join(path_split[0], 'bi_mask', path_split[1])
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_0 = mask[:, :, 0]
    mask_1 = mask[:, :, 1]
    mask_2 = mask[:, :, 2]
    assert (mask_0 == mask_1).all() == True
    assert (mask_0 == mask_2).all() == True
    mask = mask_0
    mask[mask>0] = 1
    output = mask * 3
    id = [2, 1]

    for i in id:
        kernel = np.ones((21, 21), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        diff = (dilation - mask) * i
        output += diff
        mask = dilation

    if not osp.exists(osp.dirname(new_path)):
        os.makedirs(osp.dirname(new_path))
    cv2.imwrite(new_path, output)
"""