import json

import torch
import numpy as np


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


class WeightInitializer(json.JSONEncoder):
    def __init__(self, name=None, fn=None, scaling=None,
                 *args, **kwargs):
        assert (name is not None) ^ (fn is not None), \
            'Exactly one of name or fn needs to be passed'

        self._name = name
        self._fn = fn

        if self._name is not None:
            assert self.init_fn is not None, \
                'Unknown init function {}'.format(self.name)
        if self._fn is not None:
            assert callable(fn)

        self._scaling = scaling
        self._args = args
        self._kwargs = kwargs

    @property
    def init_fn(self):
        if self._fn is not None:
            return self._fn
        if self._name is not None:
            return {
                'uniform': torch.nn.init.uniform_,
                'normal': torch.nn.init.normal_,
                'constant': torch.nn.init.constant_,
                'ones': torch.nn.init.ones_,
                'zeros': torch.nn.init.zeros_,
                'eye': torch.nn.init.eye_,
                'dirac': torch.nn.init.dirac_,
                'xavier_uniform': torch.nn.init.xavier_uniform_,
                'xavier_normal': torch.nn.init.xavier_normal_,
                'kaiming_uniform': torch.nn.init.kaiming_uniform_,
                'kaiming_normal': torch.nn.init.kaiming_normal_,
                'orthogonal': torch.nn.init.orthogonal_,
                'sparse': torch.nn.init.sparse_
            }.get(self._name)

    def __call__(self, t):
        self.init_fn(t, *self._args, **self._kwargs)
        if self._scaling is not None:
            with torch.no_grad():
                    t.mul_(self._scaling)

    def default(self, o):
        return o.__dict__


def swish_activation(x):
    return x * torch.sigmoid(x)


"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu and torch.cuda.is_available() else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
