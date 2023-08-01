import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from wslane.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, teacher_process=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)
        self.new_mask = cfg.seg_branch if 'seg_branch' in cfg else False
        # self.new_mask = False
        self.teacher_process = Process(teacher_process, cfg) if teacher_process is not None else None

    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            if self.new_mask:
                mask_path = sample['mask_path']
                path_split = osp.split(mask_path)
                new_path = osp.join(path_split[0], 'new_mask_1', path_split[1])
                label = cv2.imread(new_path, cv2.IMREAD_UNCHANGED)
            else:
                label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})
        if self.teacher_process is not None:
            sample = self.teacher_process(sample)
        else:
            sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})
        if 'lane_exist' in data_info:
            sample.update({'number': data_info['lane_exist'].sum()})
        else:
            sample.update({'number': len(data_info['lanes'])})

        return sample

    def re_process(self, meta_batch):
        device = meta_batch['img'].device
        imgs = meta_batch['img'].cpu().numpy() * 255.
        imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.uint8)
        segs = meta_batch['seg'].cpu().numpy().astype(np.uint8)
        y_cord = np.linspace(self.cfg.img_h-1, 0, num=self.cfg.num_points)
        batch_lanes = []
        lane_line = meta_batch['lane_line'].cpu().numpy()
        for batch in lane_line:
            lanes = []
            for line in batch:
                line = line[-72:]
                line = np.concatenate((line[:,None], y_cord[:,None]), axis=1)
                lane = line[line[:, 0]>0]
                lane = np.flip(lane, axis=0)
                if len(lane) > 2:
                    lanes.append(lane.tolist())
            if len(lanes) > 0:
                batch_lanes.append(lanes)

        shape = meta_batch['lane_line'].shape
        lanes_tensor = torch.zeros(shape[0], self.cfg.max_lanes, shape[2], dtype=torch.float32, device=device)
        for idx, (img, seg, lanes) in enumerate(zip(imgs, segs, batch_lanes)):
            sample = {'img':img, 'mask':seg, 'lanes':lanes}
            sample = self.processes(sample)
            meta_batch['img'][idx, ...] = sample['img'].to(device=device)
            meta_batch['seg'][idx, ...] = sample['seg'].to(device=device, dtype=torch.int64)
            lanes_tensor[idx, ...] = torch.from_numpy(sample['lane_line']).to(device=device)
        meta_batch['lane_line'] = lanes_tensor

        return meta_batch