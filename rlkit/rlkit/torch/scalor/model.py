
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .discovery import ProposalRejectionCell
from .propagation import PropagationCell
from .modules import ImgEncoder, ZWhatEnc, GlimpseDec, BgDecoder, BgEncoder, ConvLSTMEncoder
from .common import *


class SCALORModel(nn.Module):

    def __init__(self, args):
        super(SCALORModel, self).__init__()
        self.args = args
        self.bg_what_std_bias = 0
        self.no_discovery = args.no_discovery
        if args.phase_conv_lstm:
            self.image_enc = ConvLSTMEncoder(args)
        else:
            self.image_enc = ImgEncoder(args)

        self.z_what_net = ZWhatEnc()
        self.glimpse_dec_net = GlimpseDec()

        self.propagate_cell = PropagationCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )
        if not self.args.phase_no_background:
            self.bg_enc = BgEncoder()
            self.bg_dec = BgDecoder()
            self.bg_prior_rnn = nn.GRUCell(bg_what_dim, bg_prior_rnn_hid_dim)
            self.bg_prior_net = nn.Linear(bg_prior_rnn_hid_dim, bg_what_dim * 2)

        self.proposal_rejection_cell = ProposalRejectionCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )

        if args.phase_parallel:
            self.image_enc = nn.DataParallel(self.image_enc)
            self.propagate_cell = nn.DataParallel(self.propagate_cell)
            self.bg_enc = nn.DataParallel(self.bg_enc)
            self.bg_dec = nn.DataParallel(self.bg_dec)
            self.proposal_rejection_cell = nn.DataParallel(self.proposal_rejection_cell)

        self.register_buffer('z_pres_disc_threshold', torch.tensor(0.7))
        self.register_buffer('prior_bg_mean_t1', torch.zeros(1))
        self.register_buffer('prior_bg_std_t1', torch.ones(1))
        self.register_buffer('color_t', self.args.color_t)

        self.prior_rnn_init_out = None
        self.prior_rnn_init_hid = None

        self.bg_prior_rnn_init_hid = None
        self.restart = True

    @property
    def p_bg_what_t1(self):
        return Normal(self.prior_bg_mean_t1, self.prior_bg_std_t1)

    def initial_temporal_rnn_hid(self, device):
        return torch.zeros((1, temporal_rnn_out_dim)).to(device), \
               torch.zeros((1, temporal_rnn_hid_dim)).to(device)

    def initial_prior_rnn_hid(self, device):
        if self.prior_rnn_init_out is None or self.prior_rnn_init_hid is None:
            self.prior_rnn_init_out = torch.zeros(1, prior_rnn_out_dim).to(device)
            self.prior_rnn_init_hid = torch.zeros(1, prior_rnn_hid_dim).to(device)

        return self.prior_rnn_init_out, self.prior_rnn_init_hid

    def initial_bg_prior_rnn_hid(self, device):
        if self.bg_prior_rnn_init_hid is None:
            self.bg_prior_rnn_init_hid = torch.zeros(1, bg_prior_rnn_hid_dim).to(device)

        return self.bg_prior_rnn_init_hid

    def reset(self):
        # should trigger init() on next one_step call
        if hasattr(self, "_state"):
            delattr(self, "_state") 

    def forward(self, seq, actions, eps=1e-15):
        bs = seq.size(0)
        seq_len = seq.size(1)
        device = seq.device
        self.init(seq)
        kl_z_pres_all = seq.new_zeros(bs, seq_len)
        kl_z_what_all = seq.new_zeros(bs, seq_len)
        kl_z_where_all = seq.new_zeros(bs, seq_len)
        kl_z_depth_all = seq.new_zeros(bs, seq_len)
        kl_z_bg_all = seq.new_zeros(bs, seq_len)
        log_imp_all = seq.new_zeros(bs, seq_len)
        log_like_all = seq.new_zeros(bs, seq_len)
        y_seq = seq.new_zeros(bs, seq_len, 3, img_h, img_w)
        for i in range(seq_len):
            action = actions[:, i]
            x = seq[:, i]
            # img_enc = img_enc_seq[:, i]
            kl_z_bg, kl_z_pres, kl_z_what, kl_z_where, kl_z_depth, log_imp, log_like, y, _  = self.one_step(x, action, eps=eps)
            kl_z_bg_all[:, i] = kl_z_bg 
            kl_z_pres_all[:, i] = kl_z_pres
            kl_z_what_all[:, i] = kl_z_what
            kl_z_where_all[:, i] = kl_z_where
            kl_z_depth_all[:, i] = kl_z_depth
            if not self.training and self.args.phase_nll:
                log_imp_all[:, i] = log_imp
            log_like_all[:, i] = log_like
            y_seq[:, i] = y

        # (bs, seq_len)
        counting = torch.stack(self.counting_list, dim=1)

        return y_seq, \
               log_like_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_what_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_where_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_depth_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_pres_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_bg_all.flatten(start_dim=1).mean(dim=1), \
               log_imp_all.flatten(start_dim=1).sum(dim=1), \
               counting, self.log_disc_list, self.log_prop_list, self.scalor_log_list

    def init(self, x):
        bs = x.size(0)
        device = x.device
        temporal_rnn_out_pre = x.new_zeros(bs, 1, temporal_rnn_out_dim).to(device)
        temporal_rnn_hid_pre = x.new_zeros(bs, 1, temporal_rnn_hid_dim)
        prior_rnn_out_pre = x.new_zeros(bs, 1, prior_rnn_out_dim)
        prior_rnn_hid_pre = x.new_zeros(bs, 1, prior_rnn_hid_dim)
        z_what_pre = x.new_zeros(bs, 1, z_what_dim)
        z_where_pre = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_where_bias_pre = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_depth_pre = x.new_zeros(bs, 1, z_depth_dim)
        z_pres_pre = x.new_zeros(bs, 1, z_pres_dim)
        cumsum_one_minus_z_pres_prop_pre = x.new_zeros(bs, 1, z_pres_dim)
        ids_pre = x.new_zeros(bs, 1)

        lengths = x.new_zeros(bs)
        i = 0 
        self.log_disc_list = []
        self.log_prop_list = []
        self.scalor_log_list = []
        self.counting_list = []
        bg_rnn_hid_pre = self.initial_bg_prior_rnn_hid(device).expand(bs, -1)
        self.image_enc.reset()
        self._state = {"z_what_pre": z_what_pre,
                        "bg_rnn_hid_pre": bg_rnn_hid_pre,
                        "z_where_pre": z_where_pre,
                        "z_depth_pre": z_depth_pre,
                        "z_pres_pre": z_pres_pre,
                        "temporal_rnn_out_pre": temporal_rnn_out_pre,
                        "temporal_rnn_hid_pre": temporal_rnn_hid_pre,
                        "prior_rnn_out_pre": prior_rnn_out_pre,
                        "prior_rnn_hid_pre": prior_rnn_hid_pre,
                        "cumsum_one_minus_z_pres_prop_pre": cumsum_one_minus_z_pres_prop_pre,
                        "z_where_bias_pre": z_where_bias_pre,
                        "ids_pre": ids_pre,
                        "lengths": lengths,
                        "i": 0}

    def one_step(self, x, action, img_enc=None, eps=1e-15):
        if img_enc is None: 
            img_enc = self.image_enc.one_step(x, action)
        bs = x.size(0)
        device = x.device
        if not hasattr(self,"_state"):
            self.init(x)
        z_what_pre = self._state["z_what_pre"]
        z_where_pre = self._state["z_where_pre"]
        z_depth_pre = self._state["z_depth_pre"]
        z_pres_pre = self._state["z_pres_pre"]
        ids_pre = self._state["ids_pre"]
        temporal_rnn_out_pre = self._state["temporal_rnn_out_pre"]
        temporal_rnn_hid_pre = self._state["temporal_rnn_hid_pre"]
        prior_rnn_out_pre = self._state["prior_rnn_out_pre"]
        prior_rnn_hid_pre = self._state["prior_rnn_hid_pre"]
        cumsum_one_minus_z_pres_prop_pre = self._state["cumsum_one_minus_z_pres_prop_pre"]
        z_where_bias_pre = self._state["z_where_bias_pre"]
        lengths = self._state["lengths"]
        i = self._state["i"]

        kl_z_what_prop = x.new_zeros(bs)
        kl_z_where_prop = x.new_zeros(bs)
        kl_z_depth_prop = x.new_zeros(bs)
        kl_z_pres_prop = x.new_zeros(bs)
        log_imp_prop = x.new_zeros(bs)
        log_prop = None

        n_objects_to_propagate = lengths.max()
        if n_objects_to_propagate != 0:

            max_length = int(torch.max(lengths))

            y_each_obj_prop, alpha_map_prop, importance_map_prop, z_what_prop, z_where_prop, \
            z_where_bias_prop, z_depth_prop, z_pres_prop, ids_prop, kl_z_what_prop, kl_z_where_prop, \
            kl_z_depth_prop, kl_z_pres_prop, temporal_rnn_out, temporal_rnn_hid, prior_rnn_out, prior_rnn_hid, \
            cumsum_one_minus_z_pres_prop, log_imp_prop, log_prop, representation_prop, only_prop_representation = \
                self.propagate_cell(
                    x, img_enc, temporal_rnn_out_pre, temporal_rnn_hid_pre, prior_rnn_out_pre, prior_rnn_hid_pre,
                    z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                    cumsum_one_minus_z_pres_prop_pre, ids_pre, lengths, max_length, i, no_disc=self.no_discovery, eps=eps
                )
        else:
            z_what_prop = x.new_zeros(bs, 1, z_what_dim)
            z_where_prop = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_shift_dim))
            z_where_bias_prop = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_shift_dim))
            z_depth_prop = x.new_zeros(bs, 1, z_depth_dim)
            z_pres_prop = x.new_zeros(bs, z_pres_dim)
            cumsum_one_minus_z_pres_prop = x.new_zeros(bs, 1, z_pres_dim)
            y_each_obj_prop = x.new_zeros(bs, 1, 3, img_h, img_w)
            alpha_map_prop = x.new_zeros(bs, 1, 1, img_h, img_w)
            importance_map_prop = x.new_zeros(bs, 1, 1, img_h, img_w)
            ids_prop = x.new_zeros(bs, 1)
            only_prop_representation = {"z_where": x.new_zeros(1, (z_where_scale_dim + z_where_shift_dim)), "z_what": x.new_zeros(1, z_what_dim), "z_depth": x.new_zeros(1, z_depth_dim)}

        alpha_map_prop_sum = alpha_map_prop.sum(1)
        alpha_map_prop_sum = \
            alpha_map_prop_sum + (alpha_map_prop_sum.clamp(eps, 1 - eps) - alpha_map_prop_sum).detach()
        y_each_obj_disc, alpha_map_disc, importance_map_disc, \
        z_what_disc, z_where_disc, z_where_bias_disc, z_depth_disc, \
        z_pres_disc, ids_disc, kl_z_what_disc, kl_z_where_disc, \
        kl_z_pres_disc, kl_z_depth_disc, log_imp_disc, log_disc, representation_disc = \
            self.proposal_rejection_cell(
                x, img_enc, alpha_map_prop_sum, ids_prop, lengths, i, no_disc=self.no_discovery, eps=eps
            )
        importance_map = torch.cat((importance_map_prop, importance_map_disc), dim=1)

        importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

        # (bs, 1, img_h, img_w)
        alpha_map = torch.cat((alpha_map_prop, alpha_map_disc), dim=1).sum(dim=1)

        alpha_map = alpha_map + (alpha_map.clamp(eps, 1 - eps) - alpha_map).detach()

        y_each_obj = torch.cat((y_each_obj_prop, y_each_obj_disc), dim=1)

        y_nobg = (y_each_obj.view(bs, -1, 3, img_h, img_w) * importance_map_norm).sum(dim=1)


        if i == 0:
            p_bg_what = self.p_bg_what_t1
        else:
            bg_what_pre = self._state["bg_what_pre"]
            bg_rnn_hid_pre = self._state["bg_rnn_hid_pre"]
            bg_rnn_hid_pre = self.bg_prior_rnn(bg_what_pre, bg_rnn_hid_pre)
            self._state["bg_rnn_hid_pre"] = bg_rnn_hid_pre
            # bg_rnn_hid_pre = self.layer_norm_h(bg_rnn_hid_pre)
            p_bg_what_mean_bias, p_bg_what_std = self.bg_prior_net(bg_rnn_hid_pre).chunk(2, -1)
            p_bg_what_mean = p_bg_what_mean_bias + bg_what_pre
            p_bg_what_std = F.softplus(p_bg_what_std + self.bg_what_std_bias)
            p_bg_what = Normal(p_bg_what_mean, p_bg_what_std)


        x_alpha_cat = torch.cat((x, (1 - alpha_map)), dim=1)
        # Background
        z_bg_mean, z_bg_std = self.bg_enc(x_alpha_cat)
        z_bg_std = F.softplus(z_bg_std + self.bg_what_std_bias)
        if self.args.phase_generate and i >= self.args.observe_frames:
            q_bg = p_bg_what
        else:
            q_bg = Normal(z_bg_mean, z_bg_std)
        z_bg = q_bg.rsample()
        # bg, one_minus_alpha_map = self.bg_dec(z_bg)
        bg = self.bg_dec(z_bg)

        bg_what_pre = z_bg
        self._state["bg_what_pre"] = bg_what_pre

        y = y_nobg + (1 - alpha_map) * bg

        p_x_z = Normal(y.flatten(1), self.args.sigma)
        log_like = p_x_z.log_prob(x.view(-1, 3, img_h, img_w).
                                    expand_as(y).flatten(1)).sum(-1)  # sum image dims (C, H, W)


        

        ########################################### Compute log importance ############################################
        if not self.training and self.args.phase_nll:
            # (bs, dim)
            log_imp_bg = (p_bg_what.log_prob(z_bg) - q_bg.log_prob(z_bg)).sum(1)

        ######################################## End of Compute log importance #########################################
        kl_z_bg = kl_divergence(q_bg, p_bg_what).sum(1)
        kl_z_pres = kl_z_pres_disc + kl_z_pres_prop
        kl_z_what = kl_z_what_disc + kl_z_what_prop
        kl_z_where = kl_z_where_disc + kl_z_where_prop
        kl_z_depth = kl_z_depth_disc + kl_z_depth_prop
        if not self.training and self.args.phase_nll:
            log_imp = log_imp_disc + log_imp_prop + log_imp_bg
        else:
            log_imp = None

        prior_rnn_out_init, prior_rnn_hid_init = self.initial_prior_rnn_hid(device)
        temporal_rnn_out_init, temporal_rnn_hid_init = self.initial_temporal_rnn_hid(device)

        new_prior_rnn_out_init = prior_rnn_out_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), prior_rnn_out_dim))
        new_prior_rnn_hid_init = prior_rnn_hid_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), prior_rnn_hid_dim))
        new_temporal_rnn_out_init = temporal_rnn_out_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), temporal_rnn_out_dim))
        new_temporal_rnn_hid_init = temporal_rnn_hid_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), temporal_rnn_hid_dim))

        if lengths.max() != 0:
            representation = {}
            for key in representation_prop.keys():
                z_prop = representation_prop[key]
                z_disc = representation_disc[key]
                representation[key] = torch.cat((z_prop, z_disc), dim=1) 

            z_what_prop_disc = torch.cat((z_what_prop, z_what_disc), dim=1)
            z_where_prop_disc = torch.cat((z_where_prop, z_where_disc), dim=1)
            z_where_bias_prop_disc = torch.cat((z_where_bias_prop, z_where_bias_disc), dim=1)
            z_depth_prop_disc = torch.cat((z_depth_prop, z_depth_disc), dim=1)
            z_pres_prop_disc = torch.cat((z_pres_prop, z_pres_disc), dim=1)
            z_mask_prop_disc = torch.cat((
                (z_pres_prop > 0).float(),
                (z_pres_disc > self.z_pres_disc_threshold).float()
            ), dim=1)
            temporal_rnn_out_prop_disc = torch.cat((temporal_rnn_out, new_temporal_rnn_out_init), dim=1)
            temporal_rnn_hid_prop_disc = torch.cat((temporal_rnn_hid, new_temporal_rnn_hid_init), dim=1)
            prior_rnn_out_prop_disc = torch.cat((prior_rnn_out, new_prior_rnn_out_init), dim=1)
            prior_rnn_hid_prop_disc = torch.cat((prior_rnn_hid, new_prior_rnn_hid_init), dim=1)
            cumsum_one_minus_z_pres_prop_disc = torch.cat([cumsum_one_minus_z_pres_prop,
                                                            x.new_zeros(bs, z_what_disc.size(1), z_pres_dim)],
                                                            dim=1)
            ids_prop_disc = torch.cat((ids_prop, ids_disc), dim=1)
        else:
            representation = representation_disc
            z_what_prop_disc = z_what_disc
            z_where_prop_disc = z_where_disc
            z_where_bias_prop_disc = z_where_bias_disc
            z_depth_prop_disc = z_depth_disc
            z_pres_prop_disc = z_pres_disc
            temporal_rnn_out_prop_disc = new_temporal_rnn_out_init
            temporal_rnn_hid_prop_disc = new_temporal_rnn_hid_init
            prior_rnn_out_prop_disc = new_prior_rnn_out_init
            prior_rnn_hid_prop_disc = new_prior_rnn_hid_init
            z_mask_prop_disc = (z_pres_disc > self.z_pres_disc_threshold).float()
            cumsum_one_minus_z_pres_prop_disc = x.new_zeros(bs, z_what_disc.size(1), z_pres_dim)
            ids_prop_disc = ids_disc

        num_obj_each = torch.sum(z_mask_prop_disc, dim=1)
        max_num_obj = int(torch.max(num_obj_each).item())
        if self.args.use_disc:
            final_representation = {"z_what": x.new_zeros(bs, max_num_obj, z_what_dim), 
                                    "z_where": x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim), 
                                    "z_depth": x.new_zeros(bs, max_num_obj, z_depth_dim)}
        z_what_pre = x.new_zeros(bs, max_num_obj, z_what_dim)
        z_where_pre = x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
        z_where_bias_pre = x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
        z_depth_pre = x.new_zeros(bs, max_num_obj, z_depth_dim)
        z_pres_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        z_mask_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        temporal_rnn_out_pre = x.new_zeros(bs, max_num_obj, temporal_rnn_out_dim)
        temporal_rnn_hid_pre = x.new_zeros(bs, max_num_obj, temporal_rnn_hid_dim)
        prior_rnn_out_pre = x.new_zeros(bs, max_num_obj, prior_rnn_out_dim)
        prior_rnn_hid_pre = x.new_zeros(bs, max_num_obj, prior_rnn_hid_dim)
        cumsum_one_minus_z_pres_prop_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        ids_pre = x.new_zeros(bs, max_num_obj)

        for b in range(bs):
            num_obj = int(num_obj_each[b])

            idx = z_mask_prop_disc[b].nonzero()[:, 0]
            if self.args.use_disc:
                for key in final_representation.keys():
                    final_representation[key][b, :num_obj] = representation[key][b, idx]
            z_what_pre[b, :num_obj] = z_what_prop_disc[b, idx]
            z_where_pre[b, :num_obj] = z_where_prop_disc[b, idx]
            z_where_bias_pre[b, :num_obj] = z_where_bias_prop_disc[b, idx]
            z_depth_pre[b, :num_obj] = z_depth_prop_disc[b, idx]
            z_pres_pre[b, :num_obj] = z_pres_prop_disc[b, idx]
            z_mask_pre[b, :num_obj] = z_mask_prop_disc[b, idx]
            temporal_rnn_out_pre[b, :num_obj] = temporal_rnn_out_prop_disc[b, idx]
            temporal_rnn_hid_pre[b, :num_obj] = temporal_rnn_hid_prop_disc[b, idx]
            prior_rnn_out_pre[b, :num_obj] = prior_rnn_out_prop_disc[b, idx]
            prior_rnn_hid_pre[b, :num_obj] = prior_rnn_hid_prop_disc[b, idx]
            cumsum_one_minus_z_pres_prop_pre[b, :num_obj] = cumsum_one_minus_z_pres_prop_disc[b, idx]
            ids_pre[b, :num_obj] = ids_prop_disc[b, idx]
        if not self.args.phase_do_remove_detach or self.args.global_step < self.args.remove_detach_step:
            z_what_pre = z_what_pre.detach()
            z_where_pre = z_where_pre.detach()
            z_depth_pre = z_depth_pre.detach()
            z_pres_pre = z_pres_pre.detach()
            z_mask_pre = z_mask_pre.detach()
            temporal_rnn_out_pre = temporal_rnn_out_pre.detach()
            temporal_rnn_hid_pre = temporal_rnn_hid_pre.detach()
            prior_rnn_out_pre = prior_rnn_out_pre.detach()
            prior_rnn_hid_pre = prior_rnn_hid_pre.detach()
            cumsum_one_minus_z_pres_prop_pre = cumsum_one_minus_z_pres_prop_pre.detach()
            z_where_bias_pre = z_where_bias_pre.detach()
        lengths = torch.sum(z_mask_pre, dim=(1, 2)).view(bs)
        self._state["z_what_pre"] = z_what_pre
        self._state["bg_what_pre"] = bg_what_pre
        self._state["z_where_pre"] = z_where_pre
        self._state["z_depth_pre"] = z_depth_pre
        self._state["z_pres_pre"] = z_pres_pre
        self._state["z_mask_pre"] = z_mask_pre
        self._state["temporal_rnn_out_pre"] = temporal_rnn_out_pre
        self._state["temporal_rnn_hid_pre"] = temporal_rnn_hid_pre
        self._state["prior_rnn_out_pre"] = prior_rnn_out_pre
        self._state["prior_rnn_hid_pre"] = prior_rnn_hid_pre
        self._state["cumsum_one_minus_z_pres_prop_pre"] = cumsum_one_minus_z_pres_prop_pre
        self._state["z_where_bias_pre"] = z_where_bias_pre
        self._state["lengths"] = lengths
        self._state["ids_pre"] = ids_pre
        self._state["i"] = i+1
        scalor_step_log = {}
        if self.args.log_phase:
            if ids_prop_disc.size(1) < importance_map_norm.size(1):
                ids_prop_disc = torch.cat((x.new_zeros(ids_prop_disc[:, 0:1].size()), ids_prop_disc), dim=1)
            id_color = self.color_t[ids_prop_disc.view(-1).long() % self.args.color_num]

            # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, 1, 1)
            id_color = id_color.view(bs, -1, 3, 1, 1)
            # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, img_h, img_w)
            id_color_map = (torch.cat((alpha_map_prop, alpha_map_disc), dim=1) > .3).float() * id_color
            mask_color = (id_color_map * importance_map_norm.detach()).sum(dim=1)
            x_mask_color = x - 0.7 * (alpha_map > .3).float() * (x - mask_color)
            scalor_step_log = {
                'y_each_obj': y_each_obj.cpu().detach(),
                'importance_map_norm': importance_map_norm.cpu().detach(),
                'importance_map': importance_map.cpu().detach(),
                'bg': bg.cpu().detach(),
                'alpha_map': alpha_map.cpu().detach(),
                'x_mask_color': x_mask_color.cpu().detach(),
                'mask_color': mask_color.cpu().detach(),
                'p_bg_what_mean': p_bg_what_mean.cpu().detach() if i > 0 else self.p_bg_what_t1.mean.cpu().detach(),
                'p_bg_what_std': p_bg_what_std.cpu().detach() if i > 0 else self.p_bg_what_t1.stddev.cpu().detach(),
                'z_bg_mean': z_bg_mean.cpu().detach(),
                'z_bg_std': z_bg_std.cpu().detach(),
            }
            if log_disc:
                for k, v in log_disc.items():
                    log_disc[k] = v.cpu().detach()
            if log_prop:
                for k, v in log_prop.items():
                    log_prop[k] = v.cpu().detach()
        self.log_disc_list.append(log_disc)
        self.log_prop_list.append(log_prop)
        self.scalor_log_list.append(scalor_step_log)
        self.counting_list.append(lengths)
        if self.args.use_disc:
            for key in final_representation.keys():
                final_representation[key] = final_representation[key][0]
            if final_representation["z_what"].shape[0] == 0:
                final_representation = {"z_where": x.new_zeros(1, (z_where_scale_dim + z_where_shift_dim)), "z_what": x.new_zeros(1, z_what_dim), "z_depth": x.new_zeros(1, z_depth_dim)}
        else:
            final_representation = only_prop_representation
        return kl_z_bg, kl_z_pres, kl_z_what, kl_z_where, kl_z_depth, log_imp, log_like, y, final_representation