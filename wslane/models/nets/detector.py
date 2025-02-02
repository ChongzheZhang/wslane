import torch.nn as nn
import torch

from wslane.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
        self.ws_learn = self.cfg.ws_learn if 'ws_learn' in cfg else False
    
    def get_lanes(self, output):
        return self.heads.get_lanes(output)

    def get_number(self, output):
        with torch.no_grad():
            batch_numbers = output['batch_num_lane']
            softmax = nn.Softmax(dim=1)
            batch_numbers = softmax(batch_numbers)
            batch_numbers = torch.max(batch_numbers, dim=1)[1]
        return batch_numbers

    def get_mask(self, output, img_size):
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            if img_size is not None:
                height = img_size[0] - self.cfg.cut_height
                width = img_size[1]
            else:
                height = self.cfg.ori_img_h - self.cfg.cut_height
                width = self.cfg.ori_img_w
            batch_seg = nn.functional.interpolate(output['seg'], size=(height, width), mode='bilinear',
                                                  align_corners=False)
            batch_seg = softmax(batch_seg)
            batch_seg = torch.max(batch_seg, dim=1)[1]
            if self.cfg.cut_height != 0:
                zeros_padding = torch.zeros(batch_seg.shape[0], self.cfg.cut_height, batch_seg.shape[2],
                                            dtype=torch.int64, device=batch_seg.device)
                batch_seg = torch.cat((zeros_padding, batch_seg), dim=1)

        return batch_seg


    def forward(self, batch):
        is_target = batch['is_target'] if 'is_target' in batch else False
        student_process = batch['teacher_student'] if 'teacher_student' in batch else False
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            out = self.heads(fea, batch=batch)
            if self.ws_learn and is_target:
                if student_process:
                    batch = self.heads.student_pseudo_label(out, batch)
                else:
                    batch = self.heads.generate_pseudo_label(out, batch)
            output = self.heads.loss(out, batch)
        else:
            output = self.heads(fea)

        return output

    def teacher_forward(self, batch):
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        out = self.heads(fea, batch=batch)
        meta_batch = self.heads.teacher_pseudo_label(out, batch)

        return meta_batch
