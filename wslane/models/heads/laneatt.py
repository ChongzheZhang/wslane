import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

from wslane.ops import nms
from wslane.utils.lane import Lane
from wslane.models.losses.focal_loss import FocalLoss

from wslane.models.utils.seg_decoder import SegDecoder

from ..registry import HEADS


@HEADS.register_module
class LaneATT(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64,
                 cfg=None):
        super(LaneATT, self).__init__()
        self.cfg = cfg
        backbone_nb_channels = cfg.featuremap_out_channel
        self.stride = cfg.featuremap_out_stride
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        self.num_branch = cfg.num_branch if 'num_branch' in cfg else False
        self.seg_branch = cfg.seg_branch if 'seg_branch' in cfg else False
        self.ws_learn = self.cfg.ws_learn if 'ws_learn' in cfg else False
        self.tri_loss = self.cfg.tri_loss if 'tri_loss' in cfg else False

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)
        if self.num_branch:
            self.nlane = self.cfg.pseudo_label_parameters.nlane + 1
            inter_channels = backbone_nb_channels // 4
            self.num_branch_layers = nn.Sequential(
                nn.Conv2d(backbone_nb_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, self.nlane, 1),
            )
            for layer in self.num_branch_layers:
                self.initialize_layer(layer)

        # self.inited = False

        self.anchors = self.anchors.cuda()
        self.anchor_ys = self.anchor_ys.cuda()
        self.cut_zs = self.cut_zs.cuda()
        self.cut_ys = self.cut_ys.cuda()
        self.cut_xs = self.cut_xs.cuda()
        self.invalid_mask = self.invalid_mask.cuda()

        if self.seg_branch:
            self.seg_conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
            self.seg_conv2 = nn.Conv2d(backbone_nb_channels//2, self.anchor_feat_channels, kernel_size=1)
            self.seg_conv3 = nn.Conv2d(backbone_nb_channels // 4, self.anchor_feat_channels, kernel_size=1)
            self.initialize_layer(self.seg_conv1)
            self.initialize_layer(self.seg_conv2)
            self.initialize_layer(self.seg_conv3)
            self.seg_decoder = SegDecoder(self.img_h, self.img_w, self.cfg.num_classes)
    def forward(self, x, **kwargs):
        output = {}
        param = self.cfg.train_parameters if self.training else self.cfg.test_parameters
        conf_threshold = param.conf_threshold
        nms_thres = param.nms_thres
        nms_topk = param.nms_topk
        pyramid_features = list(x[1:])
        pyramid_features.reverse()
        x = x[-1]
        batch_features = self.conv1(x)
        batch_anchor_features = self.cut_anchor_features(batch_features)
        if self.num_branch:
            batch_num_lane = self.num_forward(x)
            output['batch_num_lane'] = batch_num_lane

        if self.seg_branch:
            pyramid_features[0] = self.seg_conv1(pyramid_features[0])
            pyramid_features[1] = self.seg_conv2(pyramid_features[1])
            pyramid_features[2] = self.seg_conv3(pyramid_features[2])
            seg_features = torch.cat([
                F.interpolate(feature,
                              size=[pyramid_features[-1].shape[2], pyramid_features[-1].shape[3]],
                              mode='bilinear',
                              align_corners=False)
                for feature in pyramid_features],
                dim=1)
            seg = self.seg_decoder(seg_features)
            output['seg'] = seg


        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold)

        # Rectify
        if self.training == False:
            if self.seg_branch:
                segmentations = nn.functional.interpolate(output['seg'], size=(self.cfg.ori_img_h, self.cfg.ori_img_w), mode='bilinear')
                for idx, ((proposals, anchors, _, _), segmentation) in enumerate(zip(proposals_list, segmentations)):
                    new_scores = self.score_rectified_by_seg(proposals, segmentation)
                    mask = new_scores > conf_threshold
                    proposals_list[idx][0] = proposals[mask]
                    proposals_list[idx][1] = anchors[mask]


        output['proposals_list'] = proposals_list


        if self.training:
            if self.ws_learn:
                meta_batch = self.generate_pseudo_label(output, kwargs['batch'])
            else:
                meta_batch = kwargs['batch']
            loss_dic = {}
            return self.loss(output, meta_batch, loss_dic)
        else:
            return output

    def num_forward(self, x):
        output_ = self.num_branch_layers(x)
        mpl = nn.MaxPool2d(output_.shape[2:])
        output = mpl(output_).view(-1, self.nlane)

        return output

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold, with_scores=None):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for idx, (proposals, attention_matrix) in enumerate(zip(batch_proposals, batch_attention_matrix)):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                if with_scores is not None:
                    scores = with_scores[idx]
                else:
                    scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append([proposals[[]], self.anchors[[]], attention_matrix[[]], None])
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append([proposals, self.anchors[keep], attention_matrix, anchor_inds])

        return proposals_list

    def match_proposals_with_targets(self, proposals, targets, t_pos=15., t_neg=20.):
        # repeat proposals and targets to generate all combinations
        num_proposals = proposals.shape[0]
        num_targets = targets.shape[0]
        # pad proposals and target for the valid_offset_mask's trick
        proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
        proposals_pad[:, :-1] = proposals
        proposals = proposals_pad
        targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
        targets_pad[:, :-1] = targets
        targets = targets_pad

        proposals = torch.repeat_interleave(proposals, num_targets,
                                            dim=0)  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] (1000,78)->(4000,78)

        targets = torch.cat(num_proposals * [targets])  # applying this 2 times on [c, d] gives [c, d, c, d] (4,78)->(4000,78)

        # get start and the intersection of offsets
        targets_starts = targets[:, 2] * self.n_strips #start y
        proposals_starts = proposals[:, 2] * self.n_strips
        starts = torch.max(targets_starts, proposals_starts).round().long()
        ends = (targets_starts + targets[:, 4] - 1).round().long()
        lengths = ends - starts + 1
        ends[lengths < 0] = starts[lengths < 0] - 1
        lengths[lengths < 0] = 0  # a negative number here means no intersection, thus zero lenght

        # generate valid offsets mask, which works like this:
        #   start with mask [0, 0, 0, 0, 0]
        #   suppose start = 1
        #   lenght = 2
        valid_offsets_mask = targets.new_zeros(targets.shape)
        all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
        #   put a one on index `start`, giving [0, 1, 0, 0, 0]
        valid_offsets_mask[all_indices, 5 + starts] = 1.
        valid_offsets_mask[all_indices, 5 + ends + 1] -= 1.
        #   put a -1 on the `end` index, giving [0, 1, 0, -1, 0]
        #   if lenght is zero, the previous line would put a one where it shouldnt be.
        #   this -=1 (instead of =-1) fixes this
        #   the cumsum gives [0, 1, 1, 0, 0], the correct mask for the offsets
        valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0.
        invalid_offsets_mask = ~valid_offsets_mask

        # compute distances
        # this compares [ac, ad, bc, bd], i.e., all combinations
        distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / (lengths.float() + 1e-9
                                                                                                )  # avoid division by zero
        INFINITY = 987654.
        distances[lengths == 0] = INFINITY
        invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
        distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

        positives = distances.min(dim=1)[0] < t_pos
        negatives = distances.min(dim=1)[0] > t_neg

        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            target_positives_indices = distances[positives].argmin(dim=1)
        invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

        return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices

    def loss(self, output, meta_batch, loss_dic):
        loss_dic = self.detection_branch_loss(output, meta_batch, loss_dic)
        if self.num_branch:
            loss_dic = self.num_branch_loss(output, meta_batch, loss_dic)
        if self.seg_branch:
            loss_dic = self.seg_branch_loss(output, meta_batch, loss_dic)

        return loss_dic

    def detection_branch_loss(self, output, meta_batch, loss_dic, cls_loss_weight=1):
        loss_dic['loss'] = 0
        loss_dic['loss_stats'] = {}

        targets = meta_batch['lane_line']
        proposals_list = output['proposals_list']
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        cls_loss = 0
        reg_loss = 0
        tri_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = self.match_proposals_with_targets(
                    anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

            # triplet loss
            if self.tri_loss:
                negative_target = meta_batch['negative_target']
                while negative_target.shape[0] < reg_target.shape[0]:
                    negative_target = torch.cat([negative_target, negative_target], dim=0)
                negative_target = negative_target[:reg_target.shape[0], 5:] / self.img_w
                tri_reg_pred = reg_pred[:, 1:] / self.img_w
                tri_reg_target = reg_target[:, 1:] / self.img_w
                tri_loss += triplet_loss(tri_reg_pred, tri_reg_target, negative_target) * 50.


        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs
        tri_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss + tri_loss
        loss_dic['loss'] = loss
        loss_dic['loss_stats']['loss'] = loss_dic['loss']
        loss_dic['loss_stats']['cls_loss'] = cls_loss
        loss_dic['loss_stats']['reg_loss'] = reg_loss
        if tri_loss:
            loss_dic['loss_stats']['tri_loss'] = tri_loss
        loss_dic['loss_stats']['batch_positives'] = total_positives

        if 'positive_pseudo_label_len' in meta_batch:
            positive_pseudo_label_len = meta_batch['positive_pseudo_label_len']
            number_gt = meta_batch['number'].view(-1, 1)
            mse_loss = nn.MSELoss()
            number_lane_loss = mse_loss(positive_pseudo_label_len, number_gt.to(dtype=torch.float32))

            loss_dic['loss'] += number_lane_loss
            loss_dic['loss_stats']['loss'] = loss_dic['loss']
            loss_dic['loss_stats']['number_lane_loss'] = number_lane_loss


        return  loss_dic

    def num_branch_loss(self, output, meta_batch, loss_dic):
        targets = meta_batch['number']
        predictions = output['batch_num_lane']
        ce_loss = nn.CrossEntropyLoss()
        number_lane_loss = ce_loss(predictions, targets)
        loss_dic['loss'] += number_lane_loss
        loss_dic['loss_stats']['loss'] = loss_dic['loss']
        loss_dic['loss_stats']['num_branch_loss'] = number_lane_loss

        return loss_dic

    def seg_branch_loss(self, output, meta_batch, loss_dic, seg_loss_weight=1):
        targets = meta_batch['seg']
        # predictions = output['seg']
        # ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0, 1.5, 2.0]).to(device=predictions.device))
        # seg_loss = ce_loss(predictions, targets) * seg_loss_weight

        # nllloss version
        predictions = F.log_softmax(output['seg'], dim=1)
        nllloss = nn.NLLLoss(weight=torch.tensor([0.5, 1.0, 1.5, 2.0]).to(device=predictions.device))
        seg_loss = nllloss(predictions, targets.long()) * seg_loss_weight
        loss_dic['loss'] += seg_loss
        loss_dic['loss_stats']['loss'] = loss_dic['loss']
        loss_dic['loss_stats']['seg_loss'] = seg_loss

        return loss_dic

    def generate_pseudo_label(self, prediction_batch, meta_batch):
        softmax = nn.Softmax(dim=1)
        param = self.cfg.pseudo_label_parameters
        conf_threshold = param.conf_threshold
        max_lanes = param.max_lanes
        predictions = prediction_batch['proposals_list']
        segmentations = nn.functional.interpolate(prediction_batch['seg'], size=(self.cfg.ori_img_h, self.cfg.ori_img_w), mode='bilinear')

        p_len = predictions[0][0].shape[1]
        max_lane = max([t[0].shape[0] for t in predictions])
        pseudo_label = torch.ones((meta_batch['img'].shape[0], max_lane, p_len), dtype=torch.float32, device=predictions[0][0].device) * -1e5
        positive_pseudo_label_len = torch.zeros((meta_batch['img'].shape[0], 1), device=predictions[0][0].device)
        pseudo_label[:, :, 0] = 1
        pseudo_label[:, :, 1] = 0
        n_lane = 0
        for i, ((proposals, _, _, _), segmentation) in enumerate(zip(predictions, segmentations)):
            if self.tri_loss:
                new_proposals = proposals[:30, :].detach().clone()
                new_scores = self.score_rectified_by_seg(new_proposals, segmentation)
                _, negative_indice = torch.sort(new_scores)
                negative_target = new_proposals[negative_indice, :]
                meta_batch['negative_target'] = negative_target[:20, :]

            scores = softmax(proposals[:, :2])[:, 1]
            mask = scores > conf_threshold
            proposals = proposals[mask]
            pre_length = proposals[:, 2] * self.n_strips + proposals[:, 4]
            valid_length = pre_length < self.n_strips
            proposals = proposals[valid_length]
            finial_scores = softmax(proposals[:, :2])[:, 1]
            positive_pseudo_label_len[i, 0] = finial_scores.sum()


            with torch.no_grad():
                proposals_copy = proposals.clone()
                proposals_copy[:, 0] = 0
                proposals_copy[:, 1] = 1
                proposals_copy[:, 2:4] = torch.maximum(proposals_copy[:, 2:4], torch.zeros_like(proposals_copy[:, 2:4]))
                proposals_copy[:, 2:4] = torch.minimum(proposals_copy[:, 2:4], torch.ones_like(proposals_copy[:, 2:4]))
                proposals_copy[:, 3] = proposals_copy[:, 3] * self.img_w
                proposals_copy[:, 4] = torch.round(proposals_copy[:, 4])
                all_lane_length = torch.round(proposals_copy[:, 2] * self.n_strips) + proposals_copy[:, 4]
                for k, lane_length in enumerate(all_lane_length.to(torch.int)):
                    proposals_copy[k, 5 + lane_length.item():] = -1e5

                this_lane = proposals_copy.shape[0]
                pseudo_label[i, :this_lane, :] = proposals_copy
                if this_lane > n_lane:
                    n_lane = this_lane

        print(f"n_lane: {n_lane}")
        n_lane = max([n_lane, 4])
        if n_lane > max_lanes:
            n_lane = max_lanes
        pseudo_label = pseudo_label[:, :n_lane, :]

        meta_batch['lane_line'] = pseudo_label
        meta_batch['positive_pseudo_label_len'] = positive_pseudo_label_len

        if self.seg_branch:
            seg_preds = prediction_batch['seg']
            seg_scores = softmax(seg_preds)
            label3 = torch.zeros_like(seg_scores[:, 0, :, :], dtype=torch.long, device=seg_scores.device)
            label3[seg_scores[:, 3, :, :]>0.5] = 3
            re_seg_scores = softmax(seg_preds[:, :3, :, :])
            other_label = torch.max(re_seg_scores, dim=1)[1]
            pseudo_seg_label = torch.maximum(label3, other_label)
            meta_batch['seg'] = pseudo_seg_label

        return meta_batch

    def score_rectified_by_seg(self, proposals, segmentation):
        with torch.no_grad():
            softmax_p = nn.Softmax(dim=1)
            softmax_s = nn.Softmax(dim=0)
            scores = softmax_p(proposals[:, :2])[:, 1]
            new_scores = torch.zeros_like(scores, device=scores.device)
            # vie_seg = segmentation.cpu().numpy()
            # vie_seg_after_softmax = softmax_p(segmentation).cpu().numpy()
            # vie_seg_after_max = torch.max(softmax_p(segmentation), dim=0)[1].cpu().numpy()
            seg = torch.max(softmax_s(segmentation), dim=0)[1]
            proposals[:, :2] = softmax_p(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                return scores
            else:
                preds = self.proposals_to_pred(proposals, get_points=True)
                for i, pred in enumerate(preds):
                    x_coords = pred[:, 0] - 1
                    y_coords = pred[:, 1] - 1
                    all_points = seg[y_coords, x_coords]
                    if len(all_points) != 0:
                        alpha = torch.sum(all_points).to(torch.float) / (len(all_points) * 3.)
                        if alpha > 0.8:
                            new_scores[i] = scores[i]
                        elif alpha < 0.3:
                            if self.training:
                                alpha = alpha - 1.3
                                new_scores[i] = scores[i] * alpha
                            else:
                                continue
                        else:
                            if self.training:
                                alpha = (0.8 - alpha) / (0.8 - 0.3)
                                new_scores[i] = scores[i] * (1 - alpha)
                            else:
                                new_scores[i] = scores[i]
                    else:
                        new_scores[i] = scores[i]
        return new_scores



    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals, get_points=False):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        point_lane = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            if get_points:
                lane_ys = lane_ys[lane_xs <= 1]
                lane_xs = lane_xs[lane_xs <= 1]
                lane_xs = torch.round(torch.mul(lane_xs, self.cfg.ori_img_w)).to(dtype=torch.long)
                lane_ys = torch.round(torch.mul(lane_ys, self.cfg.ori_img_h)).to(dtype=torch.long)
                points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
                point_lane.append(points)
            else:
                points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
                lane = Lane(points=points.cpu().numpy(),
                            metadata={
                                'start_x': lane[3],
                                'start_y': lane[2],
                                'conf': lane[1]
                            })
                lanes.append(lane)
        if get_points:
            return point_lane
        else:
            return lanes

    def get_lanes(self, output, as_lanes=True):
        proposals_list = output['proposals_list']
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded
