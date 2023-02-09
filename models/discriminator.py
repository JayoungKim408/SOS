###########################################################################
## Copyright (C) 2023 Samsung SDS Co., Ltd. All rights reserved.
## Released under the Samsung SDS source code license.
## For details on the scope of licenses, please refer to the License.md file 
## (https://github.com/JayoungKim408/SOS/License.md).
###########################################################################


from . import utils, layers
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
