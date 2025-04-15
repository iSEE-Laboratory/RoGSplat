from __future__ import print_function, division

import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from lib.human_loader_rp import HumanDataset
# from lib.human_loader import HumanDataset
from lib.network_2 import HumanModel

from config.config import Config as config
from lib.train_recoder import Logger, file_backup
from lib.loss import l1_loss, l2_loss, ssim, psnr
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from lib.utils import set_requires_grad

warnings.filterwarnings("ignore", category=UserWarning)
import lpips

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = HumanModel(self.cfg)
        self.model.net1.load_state_dict(torch.load(cfg.stage1_ckpt, map_location='cuda')['network'], strict=True)
        self.model.net1.requires_grad_(False)

        self.train_set = HumanDataset(self.cfg.dataset, phase='train')
        num_workers = 0 if args.debug else self.cfg.batch_size * 2
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=True)  #
        self.train_iterator = iter(self.train_loader)
        self.val_set = HumanDataset(self.cfg.dataset, phase='val')
        num_workers = 0 if args.debug else 2
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
        self.len_val = int(len(self.val_loader))  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr,
                                                       self.cfg.stage_2_num_steps+self.cfg.depth_refine_num_steps+100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()

        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.depth_ckpt:
            state_dict={}
            depth_model=torch.load(self.cfg.depth_ckpt, map_location='cuda')['network']
            for k,v in depth_model.items():
                if k.startswith('depth_refiner'):
                    state_dict[k[14:]]=v
            self.model.depth_refiner.load_state_dict(state_dict)

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
        set_requires_grad(self.loss_fn_vgg, False)

    def train_depth_refine(self):
        self.model.train()

        progress_bar = tqdm(range(self.total_steps, self.cfg.depth_refine_num_steps))
        for _ in progress_bar:
            self.optimizer.zero_grad()

            data = self.fetch_data(phase='train')
            depth_loss = self.model(data, render=False)

            loss =depth_loss[0]
            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics = {
                'refine': depth_loss[0].item(),
            }
            self.logger.push(metrics)
            progress_bar.set_postfix(loss=loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            self.total_steps += 1

        self.save_ckpt(save_path=Path('%s/%s_depth_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
        self.total_steps = 0

    def train(self):
        self.model.train()

        progress_bar = tqdm(range(self.total_steps, self.cfg.stage_2_num_steps))
        for _ in progress_bar:
            self.optimizer.zero_grad()

            data = self.fetch_data(phase='train')
            img_pred, mask_pred, depth_loss = self.model(data,
                                                         True)

            # Loss
            img_gt = data['target']['image'].cuda()
            mask_gt = data['target']['mask'].cuda()

            Ll1 = l1_loss(img_pred, img_gt)

            Lssim = 1.0 - ssim(img_pred, img_gt)

            # Llpips=self.loss_fn_vgg(img_gt*2-1,img_pred*2-1)
            loss = 0.8 * Ll1 + 0.2 * Lssim + 0.1*depth_loss[0] #+0.1*Llpips

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics = {
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
            }
            self.logger.push(metrics)
            progress_bar.set_postfix(loss=loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                img_pred = self.model(data, True)[0]
                img_gt = data['target']['image'].cuda()

                psnr_value = psnr(img_pred, img_gt).mean().double()
                psnr_list.append(psnr_value.item())
                tmp_novel = img_pred[0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = '%s/%s_%s.jpg' % (
                    cfg.record.show_path, self.total_steps, data['target']['sample_name'][0])
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")
        self.logger.write_dict({'val_psnr': val_psnr}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)
        for view in ['ref']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/config_rp.yaml")
    # cfg.load("config/config.yaml")
    cfg = cfg.get_cfg()
    cfg.defrost()

    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(3407)
    np.random.seed(3407)

    trainer = Trainer(cfg)
    # if cfg.depth_ckpt is None:
    # trainer.train_depth_refine()
    trainer.train()
