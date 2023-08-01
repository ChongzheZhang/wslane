import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from tqdm import tqdm, trange
from wslane.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from wslane.datasets import build_dataloader
from wslane.utils.recorder import build_recorder
from wslane.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None
        self.eval_loader = None
        self.num_branch = self.cfg.num_branch if 'num_branch' in self.cfg else False
        self.seg_branch = self.cfg.seg_branch if 'seg_branch' in self.cfg else False
        self.ws_combine_learn = self.cfg.ws_combine_learn if 'ws_combine_learn' in self.cfg else False
        self.teacher_process = self.cfg.kd_learn if 'kd_learn' in self.cfg else False
        self.ema = self.cfg.ema if 'ema' in self.cfg else False
        if self.teacher_process and self.ema:
            self.teacher_net  = build_net(self.cfg)
            self.teacher_net = MMDataParallel(self.teacher_net, device_ids=range(self.cfg.gpus)).cuda()
            self.ema_rate = self.cfg.ema_rate if 'ema_rate' in self.cfg else 0.99
            self.teacher_net = self.update_ema_variables(self.teacher_net, self.net, self.ema_rate, 0)

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader, target_loader=None):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            data.update({'is_target': True})
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            lr = self.optimizer.param_groups[0]['lr']
            self.recorder.lr = lr
            self.recorder.record_train()
            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                self.recorder.record('train')

    def wslearn_epoch(self, epoch, train_loader, source_loader):
        self.net.train()
        if self.teacher_process and self.ema:
            self.teacher_net.train()
        end = time.time()
        assert len(train_loader) == len(source_loader)
        max_iter = len(train_loader)
        targetloader_iter = enumerate(train_loader)
        sourceloader_iter = enumerate(source_loader)
        for i in range(max_iter):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            _, source_data = sourceloader_iter.__next__()
            source_data = self.to_cuda(source_data)
            source_output = self.net(source_data)
            self.optimizer.zero_grad()
            loss = source_output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            del loss
            for key in list(source_output['loss_stats']):
                if 'loss' in key:
                    source_output['loss_stats'][f'source_{key}'] = source_output['loss_stats'][key]
                del source_output['loss_stats'][key]
            self.recorder.update_loss_stats(source_output['loss_stats'])

            _, target_data = targetloader_iter.__next__()
            target_data.update({'is_target': True})
            target_data = self.to_cuda(target_data)
            if self.teacher_process:
                if self.ema:
                    target_data = self.teacher_net.module.teacher_forward(target_data)
                else:
                    target_data = self.net.module.teacher_forward(target_data)
                target_data = train_loader.dataset.re_process(target_data)
                target_data.update({'teacher_student': True})
            output = self.net(target_data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if self.ema:
                self.teacher_net = self.update_ema_variables(self.teacher_net, self.net, self.ema_rate, i+1)
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            lr = self.optimizer.param_groups[0]['lr']
            self.recorder.lr = lr
            self.recorder.record_train()
            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        if self.ws_combine_learn:
            source_loader = build_dataloader(self.cfg.dataset.source, self.cfg, is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 1
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net, self.optimizer, self.scheduler, self.recorder)
        for epoch in trange(start_epoch, self.cfg.epochs + 1, initial=start_epoch, total=self.cfg.epochs):
            self.recorder.epoch = epoch
            if self.ws_combine_learn:
                self.wslearn_epoch(epoch, train_loader, source_loader)
            else:
                self.train_epoch(epoch, train_loader)
            if epoch >= self.cfg.eval_from or epoch % self.cfg.eval_ep == 0:
                self.save_ckpt()
                self.test(on_val=True)
            if epoch == self.cfg.epochs:
                self.save_ckpt()
                self.test()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def test(self, on_val=False, other_data=None):
        name = None
        if other_data is not None:
            name = other_data
            if other_data == 'debug':
                self.eval_loader = build_dataloader(self.cfg.dataset.debug, self.cfg, is_train=False)
            elif other_data == 'porsche':
                self.eval_loader = build_dataloader(self.cfg.dataset.porsche, self.cfg, is_train=False)
        else:
            if on_val:
                name = "validation"
                if not self.val_loader:
                    self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
                self.eval_loader = self.val_loader
            else:
                name = "test"
                if not self.test_loader:
                    self.test_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
                self.eval_loader = self.test_loader
        self.net.eval()
        lane_preds = []
        num_preds = []
        pred_dict = {}
        ntosave = 200
        interval = len(self.eval_loader.dataset.data_infos) // ntosave if len(
            self.eval_loader.dataset.data_infos) > ntosave else 1
        for idx, data in enumerate(tqdm(self.eval_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                lane_pred = self.net.module.heads.get_lanes(output)
                pred_dict['lane_pred'] = lane_pred
                lane_preds.extend(lane_pred)
                has_num_branch = self.cfg.num_branch if self.cfg.haskey('num_branch') else False
                if has_num_branch:
                    num_pred = self.net.module.get_number(output)
                    num_preds.append(num_pred)
                    
            if self.cfg.view:
                if (idx + 1) % interval == 0 and (idx + 1) // interval <= ntosave:
                    if self.num_branch:
                        num_pred = self.net.module.get_number(output)
                        pred_dict['num_pred'] = num_pred
                    if self.seg_branch:
                        if 'img_size' in self.eval_loader.dataset.data_infos[idx]:
                            img_size = self.eval_loader.dataset.data_infos[idx]['img_size'][:-1]
                        else:
                            img_size = None
                        seg_pred = self.net.module.get_mask(output, img_size)
                        pred_dict['seg_pred'] = seg_pred.cpu().numpy()
                    if hasattr(self.eval_loader.dataset, 'new_view'):
                        self.eval_loader.dataset.new_view(pred_dict, idx)
                    else:
                        self.eval_loader.dataset.view(lane_pred, data['meta'])

        self.recorder.logger.info('Result in {} dataset'.format(name))
        out = self.eval_loader.dataset.evaluate(lane_preds, self.cfg.work_dir)
        if isinstance(out, dict):
            if "F1" in out:
                self.recorder.logger.info(out['F1'])
                metric = out['F1']
        else:
            self.recorder.logger.info(out)
            metric = out

        if on_val:
            self.recorder.record_eval(out)
            if metric > self.metric:
                self.metric = metric
                self.save_ckpt(is_best=True)
            self.recorder.logger.info('Best metric: ' + str(self.metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        lane_preds = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                lane_preds.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(lane_preds,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)

    def update_ema_variables(self, ema_model, model, alpha_teacher, iteration):
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model
