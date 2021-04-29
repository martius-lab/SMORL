# -*- coding: future_fstrings -*-

import argparse
import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from .log_utils import log_summary
from .utils import save_ckpt, load_ckpt, print_scalor
from .common import *

from tensorboardX import SummaryWriter

import errno
import os
import torch
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from .model import SCALORModel


class SCALOR(object):
    """
    A class for learning object-oriented representations from sequenses
    """

    def __init__(self, 
                 n_itr,
                 batch_size,
                 lr, 
                 num_cell_h, 
                 num_cell_w, 
                 max_num_obj,
                 explained_ratio_threshold, 
                 ratio_anc, 
                 sigma, 
                 size_anc, 
                 var_anc, 
                 var_s,  
                 z_pres_anneal_end_value,
                 use_disc=False,
                 logdir="./results/",
                **kwargs):
        cfg.summary_dir = os.path.join(logdir, "summary/events")
        try:
            os.makedirs(cfg.summary_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass      
        cfg.ckpt_dir = os.path.join(logdir, "checkpoints")
        try:
            os.makedirs(cfg.ckpt_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass  
        cfg.use_disc = use_disc
        cfg.batch_size = batch_size  
        cfg.lr = lr 
        cfg.num_cell_h = num_cell_h 
        cfg.num_cell_w = num_cell_w 
        cfg.max_num_obj = max_num_obj
        cfg.explained_ratio_threshold = explained_ratio_threshold 
        cfg.ratio_anc = ratio_anc 
        cfg.sigma = sigma 
        cfg.size_anc = size_anc 
        cfg.var_anc = var_anc 
        cfg.var_s = var_s  
        cfg.z_pres_anneal_end_value = z_pres_anneal_end_value
        cfg.no_discovery = False
        cfg.log_phase = True
        cfg.global_step = 0
        cfg.n_itr = n_itr
        cfg.tau = 1.0
        cfg.seq_size = seq_len
        self.cfg = cfg


        # init from original scalor code
        self.cfg.color_t = torch.rand(700, 3)

        if not os.path.exists(self.cfg.ckpt_dir):
            os.mkdir(self.cfg.ckpt_dir)
        if not os.path.exists(self.cfg.summary_dir):
            os.mkdir(self.cfg.summary_dir)

        self.device = torch.device("cuda" if not self.cfg.nocuda and torch.cuda.is_available() else "cpu")
        self.model = SCALORModel(self.cfg)
        self.model.to(self.device)
        self.model.train()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.cfg.lr)
        self.writer = SummaryWriter(self.cfg.summary_dir)
        self.log_tau_gamma = np.log(self.cfg.tau_end) / self.cfg.tau_ep

    def encode(self, x, action):
        _, _, _, _, _, _, _, _, representation  = self.model.one_step(x, action)
        if self.model.restart:
            _, _, _, _, _, _, _, _, representation  = self.model.one_step(x, action)
            self.model.restart = False
        return representation
        
    def reset(self):
        self.model.reset()
        self.model.restart = True
    
    def train(self, imgs, actions):
        for i in range(self.cfg.n_itr):
            imgs_batch, actions_batch = self._get_batch(imgs, actions, self.cfg.batch_size, self.cfg.seq_size)
            imgs_batch = imgs_batch.reshape(self.cfg.batch_size, self.cfg.seq_size, 3, 64, 64)
            self._update_parameters(torch.Tensor(imgs_batch).to(self.device), torch.Tensor(actions_batch).to(self.device), i)

    def _get_batch(self, imgs, actions, batch_size, seq_size):

        n_episodes = imgs.shape[0]
        rollout_length = imgs.shape[1]
        img_dim = imgs.shape[2]
        actions_dim = actions.shape[2]

        imgs = imgs.reshape(n_episodes * rollout_length, img_dim)
        actions = actions.reshape(n_episodes * rollout_length, actions_dim)
        idxs_ep = np.random.randint(0, n_episodes, size=batch_size)
        in_episode = np.random.randint(0, rollout_length - seq_size, size=batch_size) 
        idxs_start = idxs_ep * rollout_length + in_episode
        idxs_end = idxs_ep * rollout_length + in_episode + seq_size
        idxs = np.asarray(list(map(range, idxs_start, idxs_end)))
        return imgs[idxs] , actions[idxs]




    # def decode(self, z):
    #     pass


    def _save_images(self, x, dec_x, itr):
        batch_size = x.size(0)
        n = min(x.shape[0], 8)
        comparison = torch.cat([x.reshape(batch_size, self.image_channels, self.image_size, self.image_size)[:n],
                                dec_x.reshape(batch_size, self.image_channels, self.image_size, self.image_size)[:n]], 0)
        path = os.path.join(self.cfg.summary_dir, str(itr)+".png")
        save_image(comparison, path, nrow=n)

    def _update_parameters(self, sample, actions, global_step):
        """code from main.py of SCALOR project"""
        log_like = 0.0 
        kl_z_what = 0.0 
        kl_z_where = 0.0 
        kl_z_depth = 0.0 
        kl_z_pres = 0.0 
        kl_z_bg = 0.0         
        total_loss = 0.0 
        stationary_background_loss = 0.0
        for k in range(4):
            sample = torch.rot90(sample, 1, dims=(3,4)) #augmentation by 90 deg rotation 

            tau = np.exp((global_step-1) * self.log_tau_gamma)
            tau = max(tau, self.cfg.tau_end)
            self.cfg.tau = tau

            log_phase = global_step % self.cfg.print_freq == 0 or global_step == 1
            self.cfg.global_step = global_step
            self.cfg.log_phase = log_phase

            imgs = sample.to(self.device)

            y_seq, log_like_, kl_z_what_, kl_z_where_, kl_z_depth_, \
            kl_z_pres_, kl_z_bg_, log_imp, counting, \
            log_disc_list, log_prop_list, scalor_log_list = self.model(imgs, actions)

            log_like += log_like_.mean(dim=0)
            kl_z_what += kl_z_what_.mean(dim=0)
            kl_z_where += kl_z_where_.mean(dim=0)
            kl_z_depth += kl_z_depth_.mean(dim=0)
            kl_z_pres += kl_z_pres_.mean(dim=0)
            kl_z_bg += kl_z_bg_.mean(0)

            total_loss += - (log_like - (kl_z_what + kl_z_where + kl_z_depth + kl_z_pres) - kl_z_bg)

        self.optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(self.model.parameters(), self.cfg.cp)
        self.optimizer.step()

        if log_phase:
            print_scalor(global_step, total_loss, log_like, kl_z_what, kl_z_where,
                            kl_z_pres, kl_z_depth)

            self.writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
            self.writer.add_scalar('train/What_KL', kl_z_what.item(), global_step=global_step)
            self.writer.add_scalar('train/Where_KL', kl_z_where.item(), global_step=global_step)
            self.writer.add_scalar('train/Pres_KL', kl_z_pres.item(), global_step=global_step)
            self.writer.add_scalar('train/Depth_KL', kl_z_depth.item(), global_step=global_step)
            self.writer.add_scalar('train/Bg_KL', kl_z_bg.item(), global_step=global_step)
            self.writer.add_scalar('train/tau', tau, global_step=global_step)

            log_summary(self.cfg, self.writer, imgs, y_seq, global_step, log_disc_list,
                        log_prop_list, scalor_log_list, prefix='train')

        if global_step % self.cfg.generate_freq == 0:
            ####################################### do generation ####################################
            self.model.eval()
            with torch.no_grad():
                self.cfg.phase_generate = True
                y_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
                kl_z_pres, kl_z_bg, log_imp, counting, \
                log_disc_list, log_prop_list, scalor_log_list = self.model(imgs, actions=actions)
                self.cfg.phase_generate = False
                log_summary(self.cfg, self.writer, imgs, y_seq, global_step, log_disc_list,
                            log_prop_list, scalor_log_list, prefix='generate')
            self.model.train()
            ####################################### end generation ####################################

        if global_step % self.cfg.save_epoch_freq == 0 or global_step == 1:
            ckpt_model_filename = f"ckpt_epoch_{global_step}.pth"
            path = os.path.join(self.cfg.ckpt_dir, ckpt_model_filename)
            self.save(path)
            # self.load(path)

    def to(self, device):
        self.model.to(device)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path, map_location=self.device)
            self.cfg.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer'])
            print("=> loaded checkpoint '{}' ".format(path))
        else:
            raise ValueError("No checkpoint!")

    def save(self, path):
        state = {
            'global_step': self.cfg.global_step,
            'state_dict': self.model.state_dict(),
            'self.optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f'{path:>2} has been successfully saved, global_step={self.cfg.global_step}')