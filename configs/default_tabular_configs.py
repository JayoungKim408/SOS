###########################################################################
## Copyright (C) 2023 Samsung SDS Co., Ltd. All rights reserved.
## Released under the Samsung SDS Public License V1.0.
## For details on the scope of licenses, please refer to the License.md file 
## (https://github.com/JayoungKim408/SOS/License.md).
##
## Code Modifications
## The configuration options of get_configs() was modified 
## according to tabular datasets.
###########################################################################

import ml_collections
import torch



def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 10000
  training.seed = 2021
  training.n_iters = 1300001
  training.snapshot_freq = 1000

  training.log_freq = 50
  training.eval_freq = 500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 100
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.1
 
  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 10.
  model.num_scales = 50  
  model.dropout = 0.1


  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  config.test = test = ml_collections.ConfigDict()

  return config