import itertools
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            optimizer_kwargs={},

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            obs_normalizer=None,
            alpha=1,

            target_q_clip=None
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        ptu.copy_model_params_from_to(self.qf1, target_qf1)
        self.target_qf1 = target_qf1
        ptu.copy_model_params_from_to(self.qf2, target_qf2)
        self.target_qf2 = target_qf2

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_qf1.parameters():
            p.requires_grad = False
        for p in self.target_qf2.parameters():
            p.requires_grad = False

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            log_alpha = ptu.ones(1).fill_(math.log(alpha))
            self.log_alpha = nn.Parameter(log_alpha)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            **optimizer_kwargs
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
            **optimizer_kwargs
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
            **optimizer_kwargs
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.target_q_clip = target_q_clip
        self.obs_normalizer = obs_normalizer
        if self.obs_normalizer is not None:
            assert self.policy.normalizer == self.obs_normalizer, \
                'Observation normalizer on SAC, but not on policy'
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs.cpu().numpy())
            # Only normalize observations for q function, as policy does
            # normalization internally
            obs_q = self.obs_normalizer.normalize(obs)
            next_obs_q = self.obs_normalizer.normalize(next_obs)
        else:
            obs_q = obs
            next_obs_q = next_obs

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        for param in itertools.chain(self.qf1.parameters(),
                                     self.qf2.parameters()):
            param.requires_grad = False

        q_new_actions = torch.min(
            self.qf1(obs_q, new_obs_actions),
            self.qf2(obs_q, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        for param in itertools.chain(self.qf1.parameters(),
                                     self.qf2.parameters()):
            param.requires_grad = True
        q1_pred = self.qf1(obs_q, actions)
        q2_pred = self.qf2(obs_q, actions)
        with torch.no_grad():
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            ent_reg = alpha * new_log_pi
            target_q_values = torch.min(
                self.target_qf1(next_obs_q, new_next_actions),
                self.target_qf2(next_obs_q, new_next_actions)
            ) - ent_reg

            if self.target_q_clip is not None:
                min_val = self.target_q_clip[0] - ent_reg
                max_val = self.target_q_clip[1] - ent_reg
                target_q_values = torch.max(torch.min(target_q_values,
                                                      max_val), min_val)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            with torch.no_grad():
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2, self.target_qf2, self.soft_target_tau
                )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        s = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
        if self.obs_normalizer is not None:
            s['obs_normalizer'] = self.obs_normalizer

        return s
