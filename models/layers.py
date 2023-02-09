# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

###########################################################################
## Copyright (C) 2023 Samsung SDS Co., Ltd. All rights reserved.
## Released under the Samsung SDS source code license.
## For details on the scope of licenses, please refer to the License.md file 
## (https://github.com/JayoungKim408/SOS/License.md).
##
## Code Modifications.
### Activation functions are added to 'get_act()': 'tanh' and 'softplus'.
### Some functions are removed:
## Dense, ddpm_conv1x1, ncsn_conv3x3, ddpm_conv3x3, CRPBlock, CondCRPBlock,
## RCUBlock, CondRCUBlock, MSFBlock, CondMSFBlock, RefineBlock, CondRefineBlock,
## ConvMeanPool, MeanPoolConv, UpsampleConv, ConditionalResidualBlock, 
## ResidualBlock, _einsum, contract_inner, NIN, AttnBlock, Upsample, Downsample,
## and ResnetBlockDDPM.
### Some functions are added from the FFJORD codebase:
## weights_init, IgnoreLinear, BlendLinear, ConcatLinear, ConcatLinear_v2,
## SquashLinear, and ConcatSquashLinear.
###########################################################################



"""Common layers for defining score networks.
"""
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_act(config):
  """Get activation functions from the config file."""

  if config.model.activation.lower() == 'elu':
    return nn.ELU()
  elif config.model.activation.lower() == 'relu':
    return nn.ReLU()
  elif config.model.activation.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.model.activation.lower() == 'swish':
    return nn.SiLU()
  elif config.model.activation.lower() == 'tanh':
    return nn.Tanh()
  elif config.model.activation.lower() == 'softplus':
    return nn.Softplus()

  else:
    raise NotImplementedError('activation function does not exist!')



def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')




###########################################################################
# Functions below are ported over from the DDIM codebase:
# https://github.com/ermongroup/ddim/blob/main/models/diffusion.py
###########################################################################

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


###########################################################################
# Functions below are ported over from the FFJORD codebase:
# https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/layers/diffeq_layers/basic.py
###########################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)
        

class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)


    def forward(self, t, x):
        return self.bn(self._layer(x))


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        out = y0 + (y1 - y0) * t[:, None]
        out = self.bn(out)
        return out


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t[:, None]
        ttx = torch.cat([tt, x], 1)

        # return self.bn(self._layer(ttx))
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(-1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
            + self._hyper_bias(t.view(-1, 1))
