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
## Released under the Samsung SDS Public License V1.0.
## For details on the scope of licenses, please refer to the License.md file 
## (https://github.com/JayoungKim408/SOS/License.md).
##
## Code Modifications.
### A module, layers.py is not imported.
### Some functions are removed:
## Combine, AttnBlockpp, Upsample, Downsample, ResnetBlockDDPMpp, 
## and ResnetBlockBigGANpp.
###########################################################################


"""Layers for defining NCSN++.
"""

import torch.nn as nn
import torch
import numpy as np


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



