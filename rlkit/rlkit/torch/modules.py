"""
Contain some self-contained modules.
"""
import torch
import torch.nn as nn
import numpy as np
import math
import rlkit.torch.pytorch_util as ptu
from einops import rearrange


class Attention(nn.Module):
    """
    Goal-dependent attention heads.
    """

    def __init__(self, embed_dim, z_goal_size, z_size,
                 num_heads=3, add_bias_kv=False, add_zero_attn=False,
                 uncond_attention=False,
                 num_uncond_queries=2,
                 num_uncond_heads=1,
                 decouple_attention_dim=False):
        super().__init__()
        self.device = ptu.device

        if decouple_attention_dim:
            self._embed_dim = embed_dim
        else:
            self._embed_dim = max(num_heads, 1) * embed_dim

        self.embedding = nn.Linear(z_size, self._embed_dim)
        torch.nn.init.zeros_(self.embedding.bias)

        if num_heads > 0:
            if decouple_attention_dim:
                # This option decouples the embedding dim from the attention
                # input/output dim. For this, we need to project the query
                # an additional time.
                attention_dim = num_heads * self._embed_dim
            else:
                attention_dim = self._embed_dim

            if self._embed_dim != attention_dim:
                self.query_proj = nn.Linear(self._embed_dim, attention_dim)
                torch.nn.init.zeros_(self.query_proj.bias)

            self.attention = nn.MultiheadAttention(attention_dim,
                                                   num_heads,
                                                   dropout=0.0,
                                                   bias=True,
                                                   add_bias_kv=add_bias_kv,
                                                   add_zero_attn=add_zero_attn,
                                                   kdim=self._embed_dim,
                                                   vdim=self._embed_dim)
            nn.init.xavier_uniform_(self.attention.out_proj.weight)
        else:
            assert uncond_attention
            self.attention = None

        if uncond_attention:
            uncond_attention_dim = self._embed_dim * num_uncond_heads
            param = torch.empty(num_uncond_queries, 1, uncond_attention_dim)
            self.uncond_queries = nn.Parameter(param)
            torch.nn.init.normal_(self.uncond_queries, std=0.02)

            self.uncond_attention = nn.MultiheadAttention(uncond_attention_dim,
                                                          num_uncond_heads,
                                                          dropout=0.0,
                                                          bias=True,
                                                          add_bias_kv=False,
                                                          add_zero_attn=False,
                                                          kdim=self._embed_dim,
                                                          vdim=self._embed_dim)
            nn.init.xavier_uniform_(self.uncond_attention.out_proj.weight)
        else:
            self.uncond_attention = None

    @property
    def output_dim(self):
        dim = 0
        if self.attention is not None:
            dim += self.attention.embed_dim
        if self.uncond_attention is not None:
            dim += len(self.uncond_queries) * self.uncond_attention.embed_dim

        return dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def embed(self, x):
        """
        Input : x (batch_size, N, z_size)
        Output: embedding (batch_size, N, embed_dim)
        """
        return self.embedding(x)

    def forward(self, state_embedding, goal_embedding, n_objects):
        """
        Input : state_embedding (batch_size, max_objects, embed_dim)
                g (batch_size, 1, embed_dim),
                n_objects (batch_size, 1)
        Output: value with shape (batch_size, embed_dim))
        """
        bs = state_embedding.shape[0]
        max_objects = state_embedding.shape[1]

        assert state_embedding.shape[-1] == self._embed_dim
        assert goal_embedding.shape[-1] == self._embed_dim

        state_embedding = state_embedding.transpose(1, 0)
        goal_embedding = goal_embedding.transpose(1, 0)

        n_objects = n_objects.to(torch.long)
        key_padding_mask = torch.zeros(bs, max_objects,
                                       device=self.device,
                                       requires_grad=False)
        key_padding_mask.scatter_(dim=1, index=n_objects, value=1)
        key_padding_mask = key_padding_mask.cumsum(1).to(torch.bool)

        if self.attention is not None:
            if self.attention.embed_dim != goal_embedding.shape[-1]:
                query = self.query_proj(goal_embedding)
            else:
                query = goal_embedding

            attn_output, _ = self.attention(query,
                                            state_embedding,
                                            state_embedding,
                                            key_padding_mask,
                                            need_weights=False)
            output = attn_output.squeeze(dim=0)
        else:
            output = None

        if self.uncond_attention is not None:
            uncond_queries = self.uncond_queries.expand(-1, bs, -1)
            uncond_attn_output, _ = self.uncond_attention(uncond_queries,
                                                          state_embedding,
                                                          state_embedding,
                                                          key_padding_mask,
                                                          need_weights=False)
            uncond_attn_output = uncond_attn_output.transpose(1, 0).reshape(bs, -1)
            if output is None:
                output = uncond_attn_output
            else:
                output = torch.cat((output, uncond_attn_output), dim=1)

        return output

    def to(self, device):
        super().to(device)
        self.device = device


def preprocess_attention_input(obs, z_size, z_goal_size, with_n_frames=None):
    n_objects = obs[:, :1]
    latent_obs = obs[:, :-(z_goal_size + 1)]
    g = obs[:, -z_goal_size:][:, None, :]

    if with_n_frames is not None:
        # obs layout:
        # 0 => n_total_objects
        # 1 : 1 + n_frames => n_objects_per_frame
        zs = latent_obs[:, 1 + with_n_frames:]

        # Add zero for z_depth and one-hot 1 for z_time_id
        g_add = torch.tensor([0, 1] + [0] * with_n_frames,
                             dtype=g.dtype, device=g.device)[None, None]
        g = torch.cat((g, g_add.expand(g.size(0), 1, -1)), dim=-1)
    else:
        # obs layout:
        # 0 => n_objects
        # 1:max_objects * z_size + 1 => objects
        # max_objects * z_size + 1 : max_objects * z_size + 2 => goal idx
        # max_objects * z_size + 2 : => goal
        zs = latent_obs[:, 1:]

        # Add zero for z_depth
        g = torch.cat((g, torch.zeros((g.size(0), 1, 1),
                                      dtype=g.dtype, device=g.device)), dim=-1)

    zs = rearrange(zs, 'b (objects z) -> b objects z', z=z_size)

    return zs, g, n_objects


class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output
