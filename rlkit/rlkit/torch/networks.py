"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm, Attention, preprocess_attention_input


def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            normalizer=None,
            last_layer_init_w=None,
            last_layer_init_b=None
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        if last_layer_init_w is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
        else:
            last_layer_init_w(self.last_fc.weight)

        if last_layer_init_b is None:
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            last_layer_init_b(self.last_fc.bias)

        self.normalizer = normalizer

    def forward(self, input, return_preactivations=False):
        if self.normalizer is not None:
            input = self.normalizer.normalize(input)

        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class AttentionMlp(nn.Module):
    def __init__(self, embed_dim, z_goal_size, z_size,
                 action_size, max_objects, hidden_sizes, output_size,
                 n_frames=None, attention_kwargs=None, **kwargs):
        super().__init__()
        self.z_goal_size = z_goal_size
        self.z_size = z_size
        self.n_frames = n_frames

        if attention_kwargs is None:
            attention_kwargs = {}

        self.attention = Attention(embed_dim,
                                   z_goal_size,
                                   z_size,
                                   **attention_kwargs)
        inp_dim = (self.attention.output_dim
                   + self.attention.embed_dim
                   + action_size)
        self.mlp = FlattenMlp(hidden_sizes, output_size, inp_dim, **kwargs)

    def forward(self, obs, actions):
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)
        goal_embedding = self.attention.embed(g)
        state_embedding = self.attention.embed(x)

        h = self.attention.forward(state_embedding, goal_embedding, n_objects)

        output = self.mlp(h, goal_embedding.squeeze(1), actions)

        return output


class DeepSetMlp(nn.Module):
    def __init__(self, key_query_size, z_goal_size, z_size, value_size,
                 action_size, max_objects, hidden_sizes, output_size,
                 embed_dim, aggregation_dim, n_frames=None, **kwargs):
        super().__init__()
        self.z_goal_size = z_goal_size
        self.z_size = z_size
        self.n_frames = n_frames

        self.embedding = nn.Linear(z_size, embed_dim)
        self.pre_aggregation = nn.Sequential(nn.Linear(embed_dim,
                                                       aggregation_dim),
                                             nn.ReLU(),
                                             nn.Linear(aggregation_dim,
                                                       aggregation_dim))
        for layer in (self.embedding,
                      self.pre_aggregation[0],
                      self.pre_aggregation[2]):
            if 'hidden_init' in kwargs:
                kwargs['hidden_init'](layer.weight)
            nn.init.zeros_(layer.bias)

        inp_dim = aggregation_dim + embed_dim + action_size
        self.mlp = FlattenMlp(hidden_sizes, output_size, inp_dim, **kwargs)

    def forward(self, obs, actions):
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)
        goal_embedding = self.embedding(g)
        state_embedding = self.embedding(x)

        h = self.pre_aggregation(state_embedding).sum(dim=1)

        output = self.mlp(h, goal_embedding.squeeze(1), actions)

        return output


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
