import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from wslane.utils.lane import Lane
from wslane.models.losses.focal_loss import FocalLoss
from wslane.models.losses.accuracy import accuracy
from wslane.ops import nms

from wslane.models.utils.roi_gather import ROIGather, LinearModule
from wslane.models.utils.seg_decoder import SegDecoder
from wslane.models.utils.dynamic_assign import assign
from wslane.models.losses.lineiou_loss import liou_loss
from ..registry import HEADS


@HEADS.register_module
class CLRHead(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 cfg=None):
        super(CLRHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.ori_img_w = cfg.ori_img_w
        self.ori_img_h = cfg.ori_img_h
        self.num_branch = cfg.num_branch if 'num_branch' in cfg else False
        self.seg_branch = cfg.seg_branch if 'seg_branch' in cfg else False
        self.ws_learn = self.cfg.ws_learn if 'ws_learn' in cfg else False
        self.tri_loss = self.cfg.tri_loss if 'tri_loss' in cfg else False
        self.pycda = self.cfg.pycda if 'pycda' in cfg else False
        self.seg_distribution_to_num = self.cfg.seg_distribution_to_num if 'seg_distribution_to_num' in cfg else False

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels

        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        # generate xys for feature map
        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.cfg.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        if self.tri_loss:
            self.tri_layers = nn.Linear(self.fc_hidden_dim*refine_layers, 12)

        weights = torch.FloatTensor(self.cfg.seg_weight)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label, weight=weights)

        if self.num_branch:
            self.nlane = self.cfg.pseudo_label_parameters.nlane + 1
            inter_channels = prior_feat_channels // 4
            self.num_branch_layers = nn.Sequential(
                nn.Conv2d(prior_feat_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, self.nlane, 1),
            )

        if self.seg_distribution_to_num:
            previous_seg_label_ave = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32), requires_grad=False)
            self.register_parameter(name='previous_seg_label_ave', param=previous_seg_label_ave)
            previous_number_of_lane_sum = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
            self.register_parameter(name='previous_number_of_lane_sum', param=previous_number_of_lane_sum)

        # init the weights here
        self.init_weights()

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        if self.tri_loss:
            for m in self.tri_layers.parameters():
                nn.init.normal_(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs] # take 36 points in a prior
        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            # put 2 priors every stride at left side
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            # put 4 priors every stride at bottom
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            # put 2 priors every stride at right side
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self, x, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        '''
        if self.training:
            self.is_target = kwargs['batch']['is_target'] if 'is_target' in kwargs['batch'] else False
        else:
            self.is_target = True
        output = {}
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors, priors_on_featmap = self.priors.repeat(batch_size, 1, 1), self.priors_on_featmap.repeat(batch_size, 1, 1)

        predictions_lists = []

        # iterative refine
        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = self.pool_prior_features(batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages, batch_features[stage], stage)

            fc_features = fc_features.view(num_priors, batch_size, -1).reshape(batch_size * num_priors, self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[1])
            if self.tri_loss and self.is_target:
                reg_features = reg_features.reshape(batch_size, -1, reg_features.shape[1])
                fc_features = fc_features.reshape(batch_size, -1, fc_features.shape[1])
                reg_features = reg_features + fc_features

                if stage == 0:
                    output['anchor_features'] = reg_features
                elif stage == (self.refine_layers - 1):
                    output['anchor_features'] = self.tri_layers(torch.cat((output['anchor_features'], reg_features), dim=-1))
                else:
                    output['anchor_features'] = torch.cat((output['anchor_features'], reg_features), dim=-1)

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits

            predictions[:, :, 2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        seg = None
        seg_features = torch.cat([
            F.interpolate(feature,
                          size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                          mode='bilinear',
                          align_corners=False)
            for feature in batch_features],
            dim=1)
        seg = self.seg_decoder(seg_features)
        output['predictions_lists'] = predictions_lists
        output['seg'] = seg
        if self.pycda:
            avgpl_1 = nn.AvgPool2d((2, 2))
            avgpl_2 = nn.AvgPool2d((4, 4))
            seg_pool_1 = avgpl_1(seg)
            seg_pool_2 = avgpl_2(seg)
            output['seg_pool_1'] = seg_pool_1
            output['seg_pool_2'] = seg_pool_2

        if self.seg_distribution_to_num and self.is_target:
            seg_scores = F.softmax(seg, dim=1)
            part_seg_scores = F.softmax(seg, dim=1)[:, :3, :, :]
            label = torch.max(part_seg_scores, dim=1)[1]
            label[seg_scores[:, 3, :, :] > 0.5] = 3
            label1_mask = label != 1
            label2_mask = label != 2
            label3_mask = label != 3
            label1_logits = seg_scores[:, 1, :, :].clone()
            label1_logits[label1_mask] = 0.
            label2_logits = seg_scores[:, 2, :, :].clone()
            label2_logits[label2_mask] = 0.
            label3_logits = seg_scores[:, 3, :, :].clone()
            label3_logits[label3_mask] = 0.
            label1_sum = torch.sum(label1_logits, dim=(2, 1)) / self.img_h
            label2_sum = torch.sum(label2_logits, dim=(2, 1)) / self.img_h
            label3_sum = torch.sum(label3_logits, dim=(2, 1)) / self.img_h

            current_seg_label_sum = torch.cat((label1_sum.view(-1, 1), label2_sum.view(-1, 1), label3_sum.view(-1, 1)),
                                              dim=1)
            output['current_seg_label_sum'] = current_seg_label_sum

        if self.num_branch and self.is_target:
            num_layer_inter = self.num_branch_layers(x[-1])
            mpl = nn.MaxPool2d(num_layer_inter.shape[2:])
            output['batch_num_lane'] = mpl(num_layer_inter).view(-1, self.nlane)


        if self.training:
            if self.ws_learn and self.is_target:
                nms_thres = self.cfg.pseudo_label_parameters.nms_thres
                output['proposals_after_nms'] = self.nms(output['predictions_lists'][-1], nms_thres)
                meta_batch = self.generate_pseudo_label(output, kwargs['batch'])
            else:
                meta_batch = kwargs['batch']
            return self.loss(output, meta_batch)
        else:
            return output

    def predictions_to_pred(self, predictions, get_points=False):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        point_lane = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            if get_points:
                lane_ys = lane_ys[lane_xs <= 1]
                lane_xs = lane_xs[lane_xs <= 1]
                lane_xs = torch.round(torch.mul(lane_xs, self.img_w)).to(dtype=torch.long)
                lane_ys = torch.round(torch.mul(lane_ys, self.img_h)).to(dtype=torch.long)
                points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
                point_lane.append(points)
            else:
                lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                           self.cfg.cut_height) / self.cfg.ori_img_h
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

    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight
        num_branch_loss_weight = self.cfg.num_branch_loss_weight if self.cfg.haskey('num_branch_loss_weight') else 1.0
        num_lane_loss_weight = self.cfg.num_lane_loss_weight if self.cfg.haskey('num_lane_loss_weight') else 1.0
        reg_loss_weight = self.cfg.reg_loss_weight if self.cfg.haskey('reg_loss_weight') else [1.0, 1.0]

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        reg_loss = 0
        # cls_acc = []
        #
        # cls_acc_stage = []
        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0,
                        self.n_strips)  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                if self.ws_learn and self.is_target:
                    reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()
                    with torch.no_grad():
                        num_positives = len(matched_row_inds)
                        positive_starts = reg_yxtl[:, 0].round().long()
                        positive_starts = torch.clamp(positive_starts, min=0, max=self.n_strips)
                        all_indices = torch.arange(num_positives, dtype=torch.long)
                        ends = (positive_starts + target_yxtl[:, -1] - 1).round().long()
                        ends = torch.clamp(ends, min=0, max=self.n_strips)
                        invalid_offsets_mask = torch.zeros((num_positives, self.n_offsets + 1), dtype=torch.int)
                        invalid_offsets_mask[all_indices, positive_starts] = 1
                        invalid_offsets_mask[all_indices, ends] -= 1
                        invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                        invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                        reg_targets[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]
                        invalid = reg_targets < 0
                        if torch.any(invalid):
                            reg_targets[invalid] = reg_pred[invalid]
                    # reg_loss = reg_loss + F.smooth_l1_loss(reg_pred, reg_targets) * reg_loss_weight[stage]
                    reg_loss = reg_loss + liou_loss(reg_pred, reg_targets, self.img_w, length=15) * reg_loss_weight[stage]
                else:
                    reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()
                    iou_loss = iou_loss + liou_loss(reg_pred, reg_targets, self.img_w, length=15)

                # calculate acc
            #     cls_accuracy = accuracy(cls_pred, cls_target)
            #     cls_acc_stage.append(cls_accuracy)
            #
            # cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1), batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        reg_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
            + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight + reg_loss

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
            }
        }
        if self.pycda and self.is_target:
            seg_pool_1_target = batch['seg_pool_1_label']
            seg_pool_2_target = batch['seg_pool_2_label']
            seg_pool_1_pred = F.log_softmax(output['seg_pool_1'], dim=1)
            seg_pool_2_pred = F.log_softmax(output['seg_pool_2'], dim=1)
            seg_pool_1_loss = self.criterion(seg_pool_1_pred, seg_pool_1_target) * seg_loss_weight
            seg_pool_2_loss = self.criterion(seg_pool_2_pred, seg_pool_2_target) * seg_loss_weight
            return_value = self.updata_loss_dic(seg_pool_1_loss, 'seg_pool_1_loss', return_value)
            return_value = self.updata_loss_dic(seg_pool_2_loss, 'seg_pool_2_loss', return_value)

        if self.seg_distribution_to_num and self.is_target:
            seg_dist_weight = self.cfg.seg_dist_weight if 'seg_dist_weight' in self.cfg else 1.0
            batch_size = batch['number'].shape[0]
            smooth_l1_loss = nn.SmoothL1Loss()
            current_num_sum = torch.sum(batch['number'])
            current_seg_label_sum = output['current_seg_label_sum']
            seg_label_sum_target = self.previous_seg_label_ave.repeat(batch_size, 1)
            for i, number in enumerate(batch['number']):
                seg_label_sum_target[i, :] *= number
            seg_distribution_loss = smooth_l1_loss(current_seg_label_sum, seg_label_sum_target) * seg_dist_weight
            return_value = self.updata_loss_dic(seg_distribution_loss, 'seg_distribution_loss', return_value)

            current_seg_label_batch_sum = torch.sum(current_seg_label_sum, dim=0, keepdim=True)
            previous_seg_label_ave = nn.Parameter((self.previous_seg_label_ave * self.previous_number_of_lane_sum + current_seg_label_batch_sum.detach()) / (
                    self.previous_number_of_lane_sum + current_num_sum),
                                                  requires_grad=False)
            self.previous_seg_label_ave = previous_seg_label_ave
            self.previous_number_of_lane_sum += current_num_sum

        if self.ws_learn and self.is_target:
            return_value['loss_stats']['reg_loss'] = reg_loss
        else:
            return_value['loss_stats']['iou_loss'] = iou_loss * iou_loss_weight

        if self.num_branch and self.is_target:
            return_value = self.num_branch_loss(output, batch, return_value, num_branch_loss_weight)

        if self.tri_loss and self.is_target:
            tri_loss = 0
            non_zero_counter = 0
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            tri_loss_weight = self.cfg.tri_loss_weight if self.cfg.haskey('tri_loss_weight') else 1.0
            for positive_target, negative_target in zip(batch['positive_target'], batch['negative_target']):
                device = positive_target.device
                positive_len = positive_target.shape[0]
                negative_len = negative_target.shape[0]
                tri_sample_len = min(positive_len, negative_len)
                positive_target = positive_target[:tri_sample_len, :]
                negative_target = negative_target[:tri_sample_len, :]
                if tri_sample_len != 0:
                    non_zero_counter += 1
                    if tri_sample_len <= 2:
                        tri_anchor = positive_target.clone()
                        positive_target = positive_target.flip(dims=[0])
                    else:
                        tri_anchor = positive_target.repeat_interleave(tri_sample_len - 1, dim=0)
                        positive_target = positive_target.repeat(tri_sample_len, 1)
                        masked_number = torch.arange(tri_sample_len, dtype=torch.long, device=device) * (tri_sample_len + 1)
                        mask = torch.ones(positive_target.shape[0], dtype=torch.bool, device=device)
                        mask[masked_number] = False
                        positive_target = positive_target[mask]
                        negative_target = negative_target[:tri_sample_len - 1, ...].repeat(tri_sample_len, 1)
                    tri_loss += triplet_loss(tri_anchor, positive_target, negative_target) * tri_loss_weight
            tri_loss /= (non_zero_counter + 1e-3)
            return_value = self.updata_loss_dic(tri_loss, 'tri_loss', return_value)

        if 'positive_pseudo_label_len' in batch:
            positive_pseudo_label_len = batch['positive_pseudo_label_len']
            number_gt = batch['number'].view(-1, 1)
            mse_loss = nn.MSELoss()
            number_lane_loss = mse_loss(positive_pseudo_label_len, number_gt.to(dtype=torch.float32)) * num_lane_loss_weight
            return_value = self.updata_loss_dic(number_lane_loss, 'number_lane_loss', return_value)

        if 'average_pseudo_number_of_lane' in batch:
            return_value['loss_stats']['ave_pseudo_lane'] = batch['average_pseudo_number_of_lane']

        # for i in range(self.refine_layers):
        #     return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

        return return_value

    def num_branch_loss(self, output, meta_batch, loss_dic, loss_weight=1.0):
        targets = meta_batch['number']
        predictions = output['batch_num_lane']
        ce_loss = nn.CrossEntropyLoss()
        num_branch_loss = ce_loss(predictions, targets) * loss_weight
        loss_dic = self.updata_loss_dic(num_branch_loss, 'num_branch_loss', loss_dic)

        return loss_dic

    def updata_loss_dic(self, loss, loss_name, loss_dic):
        loss_dic['loss'] += loss
        loss_dic['loss_stats']['loss'] = loss_dic['loss']
        loss_dic['loss_stats'][f'{loss_name}'] = loss
        return loss_dic

    def generate_pseudo_label(self, prediction_batch, meta_batch):
        softmax = nn.Softmax(dim=1)
        param = self.cfg.pseudo_label_parameters
        conf_threshold = param.conf_threshold
        max_lanes = param.max_lanes
        batch_size = meta_batch['img'].shape[0]
        device = meta_batch['img'].device
        predictions = prediction_batch['proposals_after_nms']
        # segmentations = nn.functional.interpolate(prediction_batch['seg'], size=(self.img_h, self.img_w),
        #                                           mode='bilinear', align_corners=False)
        segmentations = prediction_batch['seg']
        anchor_len = predictions[0][0].shape[1]
        pseudo_label = torch.ones(batch_size, max_lanes, anchor_len, dtype=torch.float32, device=device) * -1e5
        positive_pseudo_label_len = torch.zeros((batch_size, 1), device=device)
        pseudo_label[:, :, 0] = 1
        pseudo_label[:, :, 1] = 0
        max_lane = 0
        pseudo_number_of_lane = 0
        meta_batch['negative_target'] = []
        meta_batch['positive_target'] = []
        for i, (proposals, anchor_idx) in enumerate(predictions):
            if self.tri_loss:
                if not self.seg_branch:
                    raise ValueError("Triplet Loss or Detection Results to Segmetation Label must have Segmentation branch")
                anchor_features_tri = prediction_batch['anchor_features'][i, ...]
                new_proposals = proposals.clone()
                new_anchor_idx = anchor_idx.clone()
                segmentation = segmentations[i, ...]
                new_scores = self.score_rectified_by_seg(new_proposals, segmentation)
                _, negative_inices = torch.sort(new_scores)
                negative_len = min(10, new_scores.shape[0])
                negative_inices = negative_inices[:negative_len]
                new_anchor_idx = new_anchor_idx[negative_inices]
                negative_anchor_features = anchor_features_tri[new_anchor_idx]
                meta_batch['negative_target'].append(negative_anchor_features)

            with torch.no_grad():
                lane_num_to_keep = torch.arange(proposals.shape[0], dtype=torch.int64, device=device)

                scores = softmax(proposals[:, :2])[:, 1]
                mask = scores > conf_threshold
                proposals = proposals[mask]
                lane_num_to_keep = lane_num_to_keep[mask]

                pre_length = proposals[:, 2] * self.n_strips + proposals[:, 5]
                valid_length = pre_length < self.n_strips
                proposals = proposals[valid_length]
                lane_num_to_keep = lane_num_to_keep[valid_length]

                if proposals.shape[0] != 0:
                    proposals[:, 0] = 0
                    proposals[:, 1] = 1
                    proposals[:, 2:6] = torch.clamp(proposals[:, 2:6], min=0, max=1)
                    proposals[:, 3] = proposals[:, 3] * (self.img_w - 1)
                    proposals[:, 5] = torch.round(proposals[:, 5] * self.n_strips)
                    proposals[:, 6:] *= (self.img_w - 1)
                    all_lane_length = torch.round(proposals[:, 2] * self.n_strips) + proposals[:, 5]
                    for k, lane_length in enumerate(all_lane_length.to(torch.int)):
                        proposals[k, 5 + lane_length.item():] = -1e5

                    these_lanes = min(proposals.shape[0], max_lanes)
                    pseudo_label[i, :these_lanes, :] = proposals[:these_lanes, :]
                    pseudo_number_of_lane += these_lanes
                    if these_lanes > max_lane:
                        max_lane = these_lanes

            lane_num_proposals = predictions[i][0][lane_num_to_keep, :2]
            if lane_num_proposals.shape[0] > 0:
                num_scores = softmax(lane_num_proposals)[:, 1]
                positive_pseudo_label_len[i, 0] = num_scores.sum()
            if self.tri_loss:
                not_equal_mask = torch.all(lane_num_to_keep.unsqueeze(1) != negative_inices.unsqueeze(0), dim=1)
                lane_num_to_keep_in_tri = lane_num_to_keep[not_equal_mask]
                positive_anchor_features_idx = anchor_idx[lane_num_to_keep_in_tri]
                positive_anchor_features = anchor_features_tri[positive_anchor_features_idx]
                meta_batch['positive_target'].append(positive_anchor_features)

        pseudo_number_of_lane /= batch_size
        max_lane = max(max_lane, 4)
        pseudo_label = pseudo_label[:, :max_lane, :]

        meta_batch['lane_line'] = pseudo_label
        meta_batch['positive_pseudo_label_len'] = positive_pseudo_label_len
        meta_batch['average_pseudo_number_of_lane'] = torch.tensor(pseudo_number_of_lane)

        if self.seg_branch:
            def seg_pred_to_label(preds):
                softmax = nn.Softmax(dim=1)
                all_seg_scores = softmax(preds)
                part_seg_scores = softmax(preds[:, :3, :, :])
                label = torch.max(part_seg_scores, dim=1)[1]
                label[all_seg_scores[:, 3, :, :]>0.5] = 3
                return label

            def get_pycda_mask(whole_preds):
                softmax = nn.Softmax(dim=1)
                whole_scores = softmax(whole_preds)
                channel = whole_scores.shape[1]
                avgpl_1 = nn.AvgPool2d((2, 2))
                avgpl_2 = nn.AvgPool2d((4, 4))
                high_prob_masks_1 = []
                high_prob_masks_2 = []
                for i in range(channel):
                    mask = torch.zeros_like(whole_preds[:, 0, :, :], device=whole_scores.device)
                    mask[whole_scores[:, i, :, :] > 0.9] = 1
                    mask_1 = avgpl_1(mask)
                    mask_2 = avgpl_2(mask)
                    high_prob_masks_1.append(mask_1)
                    high_prob_masks_2.append(mask_2)
                rectify_1 = (0. + (1 - sum(high_prob_masks_1))).unsqueeze(dim=1)
                rectify_2 = (0. + (1 - sum(high_prob_masks_2))).unsqueeze(dim=1)
                rectify_scores_1 = torch.cat((rectify_1, rectify_1, rectify_1, rectify_1), dim=1)
                rectify_scores_2 = torch.cat((rectify_2, rectify_2, rectify_2, rectify_2), dim=1)
                rectify_dic = {'rectify_scores_1': rectify_scores_1, 'rectify_scores_2': rectify_scores_2}
                return rectify_dic

            with torch.no_grad():
                seg_preds = prediction_batch['seg']
                pseudo_seg_label = seg_pred_to_label(seg_preds)
                meta_batch['seg'] = pseudo_seg_label

                if self.pycda:
                    seg_pool_1 = prediction_batch['seg_pool_1']
                    seg_pool_2 = prediction_batch['seg_pool_2']
                    meta_batch['seg_pool_1_label'] = seg_pred_to_label(seg_pool_1)
                    meta_batch['seg_pool_2_label'] = seg_pred_to_label(seg_pool_2)

        return meta_batch

    def nms(self, batch_proposals, nms_thres):
        softmax = nn.Softmax(dim=1)
        proposals_after_nms = []
        for idx, proposals in enumerate(batch_proposals):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                nms_predictions = proposals.detach().clone()
                nms_predictions = torch.cat([nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
                nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
                nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)
                keep, num_to_keep, _ = nms(nms_predictions, scores, overlap=nms_thres, top_k=200)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            proposals_after_nms.append([proposals, anchor_inds])

        return proposals_after_nms

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        softmax = nn.Softmax(dim=1)

        decoded = []
        predictions_batch = output['predictions_lists'][-1]
        # if self.seg_branch and self.ws_learn:
        #     height = self.cfg.ori_img_h - self.cfg.cut_height
        #     segmentations = output['seg']
        for idx, predictions in enumerate(predictions_batch):
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            # if self.seg_branch and self.ws_learn:
            #     new_scores = self.score_rectified_by_seg(predictions, segmentations[idx, ...])
            #     mask = new_scores > threshold
            #     predictions = predictions[mask]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded

    def score_rectified_by_seg(self, proposals, segmentation):
        upper_thr = self.cfg.rectify_parameters.upper_thr
        lower_thr = self.cfg.rectify_parameters.lower_thr
        with torch.no_grad():
            softmax_p = nn.Softmax(dim=1)
            softmax_s = nn.Softmax(dim=0)
            scores = softmax_p(proposals[:, :2])[:, 1]
            new_scores = torch.zeros_like(scores, device=scores.device)
            seg = torch.max(softmax_s(segmentation), dim=0)[1]
            proposals[:, 5] = torch.round(proposals[:, 5] * self.n_strips)
            if proposals.shape[0] == 0:
                return scores
            else:
                preds = self.predictions_to_pred(proposals, get_points=True)
                for i, pred in enumerate(preds):
                    x_coords = pred[:, 0] - 1
                    y_coords = pred[:, 1] - 1
                    all_scores = seg[y_coords, x_coords]
                    if len(all_scores) != 0:
                        alpha = torch.sum(all_scores).to(torch.float) / (len(all_scores) * 3.)
                        if alpha > upper_thr:
                            new_scores[i] = scores[i]
                        elif alpha < lower_thr:
                            if self.training:
                                alpha = alpha - (1 + lower_thr)
                                new_scores[i] = scores[i] * alpha
                            else:
                                continue
                        else:
                            alpha = (upper_thr - alpha) / (upper_thr - lower_thr)
                            new_scores[i] = scores[i] * (1 - alpha)
                    else:
                        new_scores[i] = scores[i]
        return new_scores