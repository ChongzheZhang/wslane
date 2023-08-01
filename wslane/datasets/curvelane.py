import os
import os.path as osp
import json
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
import wslane.utils.curvelane_metric as curvelane_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl
import random

LIST_FILE = {
    'train': 'train/train.txt',
    'valid': 'valid/valid.txt',
    'test': 'valid/valid.txt',
    'debug': 'debug/debug.txt',
}

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)

@DATASETS.register_module
class Curvelane(BaseDataset):
    def __init__(self, data_root, split, processes=None, teacher_process=None, data_size=None, cfg=None):
        super().__init__(data_root, split, processes, teacher_process, cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.data_size = data_size
        self.num_branch = cfg.num_branch if 'num_branch' in cfg else False
        self.seg_branch = cfg.seg_branch if 'seg_branch' in cfg else False
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading Curvelane annotations...')
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/curvelane_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(len(anno['lanes']) for anno in self.data_infos)
                if self.training:
                    random.shuffle(self.data_infos)
                if self.data_size is not None:
                    self.data_infos = self.data_infos[:self.data_size]
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            lines = list_file.readlines()
            for line in tqdm(lines):
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

        # cache data infos to file
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)
        if self.training:
            random.shuffle(self.data_infos)
        if self.data_size is not None:
            self.data_infos = self.data_infos[:self.data_size]

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, self.split, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
        infos['mask_path'] = img_path.replace("images", "seg_label")[:-3] + 'png'
        img = cv2.imread(img_path)
        infos['img_size'] = img.shape

        anno_path = img_path[:-3] + 'lines.json'  # remove sufix jpg and add lines.txt
        anno_path = anno_path.replace("images", "labels")
        with open(anno_path, 'r') as anno_file:
            data = json.load(anno_file)
        lanes = []
        for line in data['Lines']:
            lane = [(float(point['x']), float(point['y'])) for point in line]
            lanes.append(lane)
        lanes = [lane for lane in lanes if len(lane) > 2]  # remove lanes with less than 2 points
        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_list(self, pred, img_size):
        img_h = img_size[0]
        img_w = img_size[1]
        ys = np.arange(img_h/2, img_h, 8) / img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_list = ([(x, y) for x, y in zip(lane_xs, lane_ys)])
            if len(lane_list) > 2:
                out.append(lane_list)

        return out

    def evaluate(self, predictions, *args):
        loss_lines = [[], [], [], []]
        print('Generating prediction output...')
        outputs = []
        for idx, pred in enumerate(tqdm(predictions)):
            img_size = self.data_infos[idx]['img_size']
            output = self.get_prediction_list(pred, img_size)
            outputs.append(output)

        result = curvelane_metric.eval_predictions(outputs, self.data_root, self.split, self.list_path,
                                                   iou_thresholds=[0.5], official=True, data_size=self.data_size)

        return result[0.5]

    def new_view(self, predictions, idx):
        data_infos = self.data_infos[idx]
        img_name = data_infos['img_name']
        img = cv2.imread(data_infos['img_path'])
        img_size = data_infos['img_size']
        img_hight = data_infos['img_size'][0]
        sample_y = range(img_hight, img_hight//2, -10)
        if self.seg_branch:
            seg = predictions['seg_pred'][0].astype('int8')
            seg = seg[:, :, np.newaxis]
            zeros = np.zeros_like(seg)
            mask_1 = np.concatenate((zeros, zeros, np.ma.array(seg, mask=(seg == 1)) * 96), axis=2)
            mask_2 = np.concatenate((zeros, np.ma.array(seg, mask=(seg == 2)) * 48, zeros), axis=2)
            mask_3 = np.concatenate((np.ma.array(seg, mask=(seg == 3)) * 32, zeros, zeros), axis=2)
            img = img + mask_1 + mask_2 + mask_3

        anno = data_infos['lanes']
        out_file = osp.join(self.cfg.work_dir, 'visualization', img_name.replace('/', '_'))
        lanes = [lane.to_array(self.cfg, flex_sample_y=sample_y, img_size=img_size) for lane in predictions['lane_pred'][0]]
        anno_array = [np.array(line) for line in anno]
        data = [(None, None, anno_array)]
        if lanes is not None:
            tp, fp, fn, ious, matches = curvelane_metric.curvelane_metric(lanes, anno, img_size)[0.5]
            assert len(matches) == len(lanes)
            data.append((matches, ious, lanes))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            for i, points in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = points.round().astype(int)
                xs, ys = points[:, 0], points[:, 1]
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color,
                                   thickness=3 if matches is None else 3)

        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)