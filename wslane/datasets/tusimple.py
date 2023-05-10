import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from wslane.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    'trainval':['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.num_branch = cfg.num_branch if 'num_branch' in cfg else False
        self.seg_branch = cfg.seg_branch if 'seg_branch' in cfg else False
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, osp.join(self.data_root, self.anno_files[0]))
        self.logger.info(result)
        result_dic = {}
        for item in json.loads(result):
            result_dic[item['name']] = item['value']
        return result_dic


    def new_view(self, predictions, idx):
        data_infos = self.data_infos[idx]
        img_name = data_infos['img_name']
        img = cv2.imread(data_infos['img_path'])
        out_file = osp.join(self.cfg.work_dir, 'visualization', img_name.replace('/', '_'))

        anno = data_infos['lanes']
        anno_array = [np.array(line) for line in anno]
        gt_lanes = []
        for gt in anno_array:
            gt_lane_x = [-2] * len(self.h_samples)
            for x, y in gt:
                k = round((y - self.h_samples[0]) / 10)
                gt_lane_x[k] = x
            gt_lanes.append(gt_lane_x)

        pred_lanes = self.pred2lanes(predictions['lane_pred'][0])
        data = [(None, anno_array)]
        if len(pred_lanes) != 0:
            matches, fp, fn  = LaneEval.bench(pred_lanes, gt_lanes, self.h_samples, 1.0, get_matches=True)
            assert isinstance(matches, list)
            assert len(matches) == len(pred_lanes)
            lanes = [lane.to_array(self.cfg) for lane in predictions['lane_pred'][0]]
            data.append((matches, lanes))
        for matches, datum in data:
            for i, points in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = points.round().astype(int)
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color,
                                   thickness=3 if matches is None else 3)

        if self.seg_branch:
            seg = predictions['seg_pred'][0].astype('int8')
            seg = seg[:, :, np.newaxis]
            zeros = np.zeros_like(seg)
            mask_1 = np.concatenate((zeros, zeros, np.ma.array(seg, mask=(seg==1))*96), axis=2)
            mask_2 = np.concatenate((zeros, np.ma.array(seg, mask=(seg==2))*48, zeros), axis=2)
            mask_3 = np.concatenate((np.ma.array(seg, mask=(seg==3))*32, zeros, zeros), axis=2)
            # mask_4 = np.concatenate((np.ma.array(seg, mask=(seg == 3)) * 24, np.ma.array(seg, mask=(seg == 3)) * 24, zeros), axis=2)
            img = img + mask_1 + mask_2 + mask_3

        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)
