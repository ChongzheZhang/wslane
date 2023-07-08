import os
import argparse
from functools import partial

import cv2
import json
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
import pickle as pkl


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(1440, 2560, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(1440, 2560, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    try:
        tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
        u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
        return np.array(splev(u, tck)).T
    except:
        return np.array([[x_cord, y_cord] for x_cord, y_cord in zip(x, y)])


def curvelane_metric(pred, anno, img_shape, width=30, iou_thresholds=[0.5], official=True):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=20) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        pred_ious = np.zeros(len(pred))
        pred_ious[row_ind] = ious[row_ind, col_ind]
        _metric[thr] = [tp, fp, fn, pred_ious, pred_ious > thr]
    return _metric


def load_curvelane_anno_data(anno_path, img_path):
    lanes = []
    with open(anno_path, 'r') as data_file:
        img_data = json.load(data_file)
    for line in img_data['Lines']:
        lane = [(float(point['x']), float(point['y'])) for point in line]
        lanes.append(lane)
    lanes = [lane for lane in lanes if len(lane) > 2]

    image = cv2.imread(img_path)
    image_shape = image.shape

    return lanes, image_shape


def load_curvelane_data(data_dir, split, file_list_path, data_size=None):
    data = []
    image_shapes = []
    cache_path = 'cache/curvelane_{}.pkl'.format(split)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            data_infos = pkl.load(cache_file)
            for data_info in data_infos:
                data.append(data_info['lanes'])
                image_shapes.append(data_info['img_size'])
    else:
        with open(file_list_path, 'r') as file_list:
            img_paths = [os.path.join(data_dir, split, line) for line in file_list.readlines()]
            filepaths = [
                os.path.join(
                    data_dir, split, line[1 if line[0] == '/' else 0:].rstrip().replace(
                        '.jpg', '.lines.json').replace('images', 'labels')) for line in file_list.readlines()
            ]

        for anno_path, img_path in tqdm(zip(filepaths, img_paths)):
            img_data, image_shape = load_curvelane_anno_data(anno_path, img_path)
            data.append(img_data)
            image_shapes.append(image_shape)

    if data_size is not None:
        data = data[:data_size]
    return data, image_shapes

def eval_predictions(pred,
                     anno_dir,
                     split,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False,
                     data_size=None):
    import logging
    logger = logging.getLogger(__name__)
    predictions = pred
    logger.info('Loading annotation data...')
    annotations, img_shape = load_curvelane_data(anno_dir, split, list_path, data_size)
    logger.info('Calculating metric for List: {}'.format(list_path))
    if sequential:
        results = t_map(
            partial(curvelane_metric,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,),
                    predictions, annotations, img_shape)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(curvelane_metric, zip(predictions, annotations, img_shape,
                        repeat(width),
                        repeat(iou_thresholds),
                        repeat(official)))

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(total_tp, total_fp,
            total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret
