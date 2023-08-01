import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
import wslane.utils.culane_metric as culane_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl
import random

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
    'debug': 'list/debug.txt',
    'porsche': 'list/porsche.txt',
    'db': 'list/db.txt',
}

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, data_root, split, processes=None, teacher_process=None, data_size=None, repeat_factor=None, cfg=None):
        super().__init__(data_root, split, processes, teacher_process, cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.data_size = data_size
        self.repeat_factor = repeat_factor
        self.num_branch = cfg.num_branch if 'num_branch' in cfg else False
        self.seg_branch = cfg.seg_branch if 'seg_branch' in cfg else False
        self.load_annotations()
        self.ori_img_w = 1640
        self.ori_img_h = 590

    def load_annotations(self):
        self.logger.info('Loading CULane annotations...')
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/culane_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in self.data_infos)
                if self.training:
                    random.shuffle(self.data_infos)
                if self.repeat_factor:
                    self.data_infos *= self.repeat_factor
                if self.data_size is not None:
                    self.data_infos = self.data_infos[:self.data_size]
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
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
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [
                list(map(float, line.split()))
                for line in anno_file.readlines()
            ]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        loss_lines = [[], [], [], []]
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)

        # if self.split == 'test':
        #     for cate, cate_file in CATEGORYS.items():
        #         result = culane_metric.eval_predictions(output_basedir,
        #                                                 self.data_root,
        #                                                 os.path.join(self.data_root, cate_file),
        #                                                 iou_thresholds=[0.5],
        #                                                 official=True)
        #
        # result = culane_metric.eval_predictions(output_basedir,
        #                                         self.data_root,
        #                                         self.list_path,
        #                                         iou_thresholds=np.linspace(0.5, 0.95, 10),
        #                                         official=True)

        result = culane_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                self.list_path,
                                                iou_thresholds=[0.5],
                                                official=True)

        return result[0.5]

    def new_view(self, predictions, idx):
        data_infos = self.data_infos[idx]
        img_name = data_infos['img_name']
        img = cv2.imread(data_infos['img_path'])
        if self.seg_branch:
            seg = predictions['seg_pred'][0].astype('int8')
            seg = seg[:, :, np.newaxis]
            zeros = np.zeros_like(seg)
            mask_1 = np.concatenate((zeros, zeros, np.ma.array(seg, mask=(seg==1))*96), axis=2)
            mask_2 = np.concatenate((zeros, np.ma.array(seg, mask=(seg==2))*48, zeros), axis=2)
            mask_3 = np.concatenate((np.ma.array(seg, mask=(seg==3))*32, zeros, zeros), axis=2)
            # mask_4 = np.concatenate((np.ma.array(seg, mask=(seg==4))*12, np.ma.array(seg, mask=(seg==4))*12, zeros), axis=2)
            img = img + mask_1 + mask_2 + mask_3

        anno = data_infos['lanes']
        out_file = osp.join(self.cfg.work_dir, 'visualization', img_name.replace('/', '_'))
        lanes = [lane.to_array(self.cfg) for lane in predictions['lane_pred'][0]]
        anno_array = [np.array(line) for line in anno]
        data = [(None, None, anno_array)]
        if lanes is not None:
            tp, fp, fn, ious, matches = culane_metric.culane_metric(lanes, anno)[0.5]
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

        # if self.num_branch:
        #     num = predictions['num_pred'][0]
        #     img = cv2.putText(img, num, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)