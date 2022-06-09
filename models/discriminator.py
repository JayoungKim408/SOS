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

from torch.nn.functional import embedding
from . import utils, layers, layerspp, normalization
import torch.nn as nn
import torch

get_act = layers.get_act
default_initializer = layers.default_init

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": nn.SiLU(),
}

@utils.register_model(name='discriminator')
class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()

    dim = config.data.image_size

    seq = []
    for item in list(config.model.dis_dims):
        seq += [
            nn.Linear(dim, item),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        ]
        dim = item
    seq += [nn.Linear(dim, 1)]
    self.seq = nn.Sequential(*seq)

  def forward(self, input):
    return self.seq(input)
