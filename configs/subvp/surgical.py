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

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_tabular_configs import get_default_configs


def get_config():
  config = get_default_configs()
  config.data.dataset = "surgical"
  config.data.image_size = 71
  config.training.batch_size = 1000
  config.eval.batch_size = 1000

  # training
  training = config.training
  training.sde = 'subvpsde'
  training.continuous = True
  training.reduce_mean = True
  training.n_iters = 10000
  training.num_fine_tuning_epochs = 5
  training.lambda_ = 0.99 
  training.angle = 70

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # model
  model = config.model
  model.beta_min = 0.5
  model.beta_max = 10.
  model.layer_type = 'concat'
  model.name = 'ncsnpp_tabular'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.activation = 'lrelu'
  model.nf = 64
  model.conditional = True
  model.hidden_dims = 512,1024,2048,1024,512
  model.embedding_type = 'positional'
  model.fourier_scale = 16

  # test
  test = config.test
  test.n_iter = 1

  # optim
  optim = config.optim
  optim.lr = 2e-04
  optim.fine_tune_lr = 2e-08

  return config
