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
### Some libraries are not imported: gc, io, time, tensorflow, 
## tensorflow_datasets and torchvision.utils.
### The make_noise function is added.
### The train function trains two SGMs, one for the non-target (or major) 
## class and the other for the target (or minor) class to oversample.
### The fine_tune function performs a post-processing procedure which further 
## enhances the oversampling quality after training the two score networks.
###########################################################################

"""Training and fine-tuning for SOS. """


import os
import math

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ncsnpp_tabular
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from torch.utils.data import DataLoader
import evaluation
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter 

FLAGS = flags.FLAGS

def make_noise(config, sde, data, shape, eps=1e-05, return_t=False, test=False):
  if test:
    t = (torch.rand(shape[0], device=config.device) * (sde.T - eps) + eps) 
  else:
    t = (torch.zeros(shape[0], device=config.device) * (sde.T - eps) + eps)

  z = torch.randn(shape, device=config.device)

  sample_idx = torch.randint(low=0, high=data.shape[0], size=(shape[0],))
  train_minor = torch.tensor(data[sample_idx]) 
  train_minor.requires_grad = True
  mean, std = sde.marginal_prob(train_minor.to(config.device), t.to(config.device))

  perturbed_data = mean + std[:, None] * z

  if return_t:  
    return perturbed_data, t
  else:
    return perturbed_data




def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  
  # Fix random seed
  randomSeed = 2021
  torch.manual_seed(randomSeed)
  torch.cuda.manual_seed(randomSeed)
  torch.cuda.manual_seed_all(randomSeed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(randomSeed)
  # tf.random.set_seed(randomSeed)

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "train-tensorboard")
  os.makedirs(tb_dir, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints") 
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Build data iterators
  train_data_total, eval_ds, (transformer, meta), major_label, minor_label, num_classes = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  batch_sizes = []
  minor_states = []

  def train_loop(label, labels, is_minor):
    train_data = torch.tensor(train_data_total[label])
    # Initialize model.
    score_model = mutils.create_model(config)
    print(score_model)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)

    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", f"checkpoint_{label}.pth")
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    batch_size = config.training.batch_size if config.training.batch_size <= len(train_data) else len(train_data)
    batch_sizes.append(batch_size)

    eval_batch_size = config.eval.batch_size if config.eval.batch_size <= len(eval_ds) else len(eval_ds)

    train_iter = iter(DataLoader(train_data, batch_size=batch_size, shuffle=True)) 
    eval_iter = iter(DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=True))

    # Setup SDEs 
    if config.training.sde.lower() == 'vpsde':
      sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
      sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
    else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    logging.info(score_model)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    if config.training.snapshot_sampling:
      sampling_shape = (batch_size, train_data.shape[1])
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters
    test_iter = config.test.n_iter

    minority = "minor" if is_minor else "major"
    logging.info(f"Starting training loop of {label}({minority}) samples at step {initial_step}.")

    max_f1 = 0
    for step in range(initial_step, num_train_steps + 1):
      try :
        batch = next(train_iter).to(config.device).float()
      except StopIteration:
        train_iter = iter(DataLoader(train_data, batch_size=batch_size, shuffle=True)) 
        batch = next(train_iter).to(config.device).float()
        pass

      loss = train_step_fn(state, batch)

      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
        writer.add_scalar(f"{minority}/{label}/training_loss", loss, step)

      # Save a temporary checkpoint to resume training after pre-emption periodically
      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
        save_checkpoint(checkpoint_meta_dir, state)

      # Report the loss on an evaluation dataset periodically
      with torch.no_grad():
        if step % config.training.eval_freq == 0:
          try:
            eval_batch = next(eval_iter).to(config.device).float()
          except StopIteration:
            eval_iter = iter(DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=True))
            eval_batch = next(eval_iter).to(config.device).float()
            pass

          eval_loss = eval_step_fn(state, eval_batch)
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
          writer.add_scalar(f"{minority}/{label}/eval_loss", eval_loss.item(), step)

          others = [i != label for i in labels]
          train_exclude = np.concatenate(train_data_total[others])
          class_num = len(train_data_total[label])

          eps=1e-05
          shape = (class_num, config.data.image_size)

          perturbed_data = make_noise(config, sde, train_exclude, shape, eps=eps,test=True)
          sample, n = sampling_fn(score_model, z=perturbed_data)

          sample = transformer.inverse_transform(sample.cpu().detach().numpy())
          train_ds_ = transformer.inverse_transform(train_exclude)
          eval_ds_ = transformer.inverse_transform(eval_ds)

          scores = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, sample]), meta, test_iter)
          logging.info(f"step: {step}, macro_f1: {scores['macro_f1'].mean()}")
          writer.add_scalar(f"{minority}/{label}/macro_f1", torch.tensor(scores["macro_f1"].mean()), step) 
          
          if max_f1 < scores["macro_f1"].mean():
            max_f1 = scores["macro_f1"].mean()
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{label}.pth'), state)
            logging.info(f"HIGHEST macro_f1 in step {step}, save checkpoint as checkpoint_{label}.pth")
    del state

  for label in major_label:
    train_loop(label, major_label+minor_label, is_minor=False)
  
  for label in minor_label:
    train_loop(label, major_label+minor_label, is_minor=True)

  # end of training and start final evaluation
  num_sampling = [ int(np.max(list(num_classes))) - list(num_classes)[i] for i in minor_label]
  # shapes = [(batch_size, config.data.image_size) for batch_size in batch_sizes]
  logging.info("Loading best score model (only for minor)")

  # Loading best score model (only for minor)
  minor_states = []
  for label in minor_label:
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{label}.pth'), state, config.device)
    logging.info(f"restore {state['step']}-th checkpoint for {label}.")

    # Setup SDEs 
    if config.training.sde.lower() == 'vpsde':
      sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
      sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
    else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    train_data = torch.tensor(train_data_total[label])
    sampling_shape = (batch_sizes[label], train_data.shape[1])
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    minor_states.append(dict( [('state', state), ('label', label), ('sde', sde), ('sampling_fn', sampling_fn)] ))

  logging.info("Compute f1 scores...")

  with torch.no_grad():
    eps=1e-05
    
    f1_score_with_N = []
    f1_score_with_Z_major = []

    train_ds_ = transformer.inverse_transform(np.concatenate(train_data_total))
    eval_ds_ = transformer.inverse_transform(eval_ds)

    for i in range(5): # repeat test 5 times 
      synthesized_minor_samples_with_N = []
      synthesized_minor_samples_with_Z_major = []

      for j, state in enumerate(minor_states):
        sde = state['sde']
        model = state['state']['model']
        sampling_fn = state['sampling_fn']

        train_exclude = np.concatenate(train_data_total[np.arange(len(train_data_total))!=j+1])
        sample_size = num_sampling[j]

        perturbed_data = make_noise(config, sde, train_exclude, shape=(sample_size, config.data.image_size), eps=eps,test=True)
        sample1, n = sampling_fn(model, z=perturbed_data) 
        sample2, n = sampling_fn(model, sample_size=sample_size) 

        sample = transformer.inverse_transform(sample1.cpu().detach().numpy())
        synthesized_minor_samples_with_Z_major.append(sample)

        sample = transformer.inverse_transform(sample2.cpu().detach().numpy())
        synthesized_minor_samples_with_N.append(sample)

      synthesized_minor_samples_with_Z_major = np.concatenate(synthesized_minor_samples_with_Z_major)
      synthesized_minor_samples_with_N = np.concatenate(synthesized_minor_samples_with_N)
      print(f"original data: {Counter(train_ds_[:, -1])}")
      print(f"oversampled data: {Counter(np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major])[:, -1])}")

      scores_with_Z_major = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major]), meta, config.test.n_iter)
      scores_with_N = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_N]), meta, config.test.n_iter)

      f1_score_with_N.append(scores_with_N['weighted_f1'])
      f1_score_with_Z_major.append(scores_with_Z_major['weighted_f1'])
      
    f1_score_with_N = np.array(f1_score_with_N) 
    f1_score_with_Z_major = np.array(f1_score_with_Z_major) 

    z_major_weighted_f1 = f1_score_with_Z_major.mean(axis=0)
    N_weighted_f1 = f1_score_with_N.mean(axis=0)

    logging.info(f"sampling from z_major: {z_major_weighted_f1}±{f1_score_with_Z_major.std(axis=0)}")
    print(f"z_major: {z_major_weighted_f1}")
    logging.info(f"sampling from z~N(0, 1): {N_weighted_f1}±{f1_score_with_N.std(axis=0)}")
    print(f"z~N(0, 1): {N_weighted_f1}")



def fine_tune(config, workdir):
  
  # Fix random seed
  randomSeed = 2021
  torch.manual_seed(randomSeed)
  torch.cuda.manual_seed(randomSeed)
  torch.cuda.manual_seed_all(randomSeed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(randomSeed)
  # tf.random.set_seed(randomSeed)

  tb_dir = os.path.join(workdir, "fine_tune-tensorboard")
  os.makedirs(tb_dir, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  checkpoint_dir = os.path.join(workdir, "checkpoints") 
  optimize_fn = losses.optimization_manager(config)

  # Build data iterators
  train_data_total, eval_ds, (transformer, meta), major_label, minor_label, num_classes = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)

  inverse_scaler = datasets.get_data_inverse_scaler(config)

  minor_states = []
  batch_sizes = []
  eval_batch_size = config.eval.batch_size if config.eval.batch_size <= len(eval_ds) else len(eval_ds)

  for label in major_label+minor_label:

    train_data = torch.tensor(train_data_total[label])
    batch_size = config.training.batch_size if config.training.batch_size <= len(train_data) else len(train_data)
    batch_sizes.append(batch_size)

    minority = "major" if label < len(major_label) else "minor"
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    state = restore_checkpoint(  os.path.join(checkpoint_dir, f'checkpoint_{label}.pth'), state, config.device)
    logging.info(f"restore {state['step']}-th checkpoint for {label}({minority}).")
    model = state['model']

    # Setup SDEs 
    if config.training.sde.lower() == 'vpsde':
      sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
      sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
    else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    sampling_shape = (batch_size, train_data.shape[1])
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    if minority == "major":
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
      major_state = dict([("label", label), ("state", state), ("score_fn", score_fn), ("sampling_fn", sampling_fn) , ('sde', sde)]   ) 
    else:
      score_fn = mutils.get_score_fn(sde, model, train=True, continuous=True)
      minor_states.append(   dict([("label", label), ("state", state), ("score_fn", score_fn), ("sampling_fn", sampling_fn), ('sde', sde)])   )



  # test before fine-tuning
  num_sampling = [ int(np.max(list(num_classes))) - list(num_classes)[i] for i in minor_label]
  shape = [(batch_size, eval_ds.shape[1]) for batch_size in batch_sizes]
  logging.info("f1 score before fine-tuning")
  [state['state']['model'].eval() for state in minor_states]

  with torch.no_grad():
    eps=1e-05
    
    f1_score_with_N = []
    f1_score_with_Z_major = []

    train_ds_ = transformer.inverse_transform(np.concatenate(train_data_total))
    eval_ds_ = transformer.inverse_transform(eval_ds)
    
    for i in range(5): # repeat test 5 times 
      synthesized_minor_samples_with_N = []
      synthesized_minor_samples_with_Z_major = []

      for j, state in enumerate(minor_states):
        sde = state['sde']
        model = state['state']['model']
        sampling_fn = state['sampling_fn']

        train_exclude = np.concatenate(train_data_total[np.arange(len(train_data_total))!=j+1])
        sample_size = num_sampling[j]

        perturbed_data = make_noise(config, sde, train_exclude, shape=(sample_size, config.data.image_size), eps=eps)
        sample1, n = sampling_fn(model, z=perturbed_data) 
        sample2, n = sampling_fn(model, sample_size=sample_size) 

        sample = transformer.inverse_transform(sample1.cpu().detach().numpy())
        synthesized_minor_samples_with_Z_major.append(sample)

        sample = transformer.inverse_transform(sample2.cpu().detach().numpy())
        synthesized_minor_samples_with_N.append(sample)

      synthesized_minor_samples_with_Z_major = np.concatenate(synthesized_minor_samples_with_Z_major)
      synthesized_minor_samples_with_N = np.concatenate(synthesized_minor_samples_with_N)
      print(f"original data: {Counter(train_ds_[:, -1])}")
      print(f"oversampled data: {Counter(np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major])[:, -1])}")

      scores_with_Z_major = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major]), meta, config.test.n_iter)
      scores_with_N = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_N]), meta, config.test.n_iter)

      f1_score_with_N.append(scores_with_N['weighted_f1'])
      f1_score_with_Z_major.append(scores_with_Z_major['weighted_f1'])
      
    f1_score_with_N = np.array(f1_score_with_N) 
    f1_score_with_Z_major = np.array(f1_score_with_Z_major) 

    z_major_weighted_f1 = f1_score_with_Z_major.mean(axis=0)
    N_weighted_f1 = f1_score_with_N.mean(axis=0)

    logging.info(f"sampling from z_major: {z_major_weighted_f1}±{f1_score_with_Z_major.std(axis=0)}")
    print(f"z_major: {z_major_weighted_f1}±{f1_score_with_Z_major.std(axis=0)}")
    logging.info(f"sampling from z~N(0, 1): {N_weighted_f1}±{f1_score_with_N.std(axis=0)}")
    print(f"z~N(0, 1): {N_weighted_f1}±{f1_score_with_N.std(axis=0)}")



  # start fine-tuning
  eps = 1e-05
  [state['state']['model'].train() for state in minor_states]

  logging.info("Starting fine-tuning loop of minor score model")
  num_fine_tuning_steps = config.training.num_fine_tuning_epochs

  for state in minor_states:
    train_iter = DataLoader(np.concatenate(train_data_total), batch_size=config.training.batch_size, shuffle=True) 

    for epoch in range(num_fine_tuning_steps):
      for i, batch in enumerate(train_iter):
        logging.info(f"start fine-tuning on {state['label']} label")
        batch = batch.to(config.device).float()
        batch.requires_grad = True
        optimizer = state['state']['optimizer']
        optimizer.zero_grad()
        sde = state['sde']
        score_fn_minor = state['score_fn']
        minor_model = state['state']['model']

        perturbed_data, t = make_noise(config, sde, batch, batch.shape, return_t=True, test=False)

        score_minor = score_fn_minor(perturbed_data, t).clone().detach()
        score_major = major_state['score_fn'](perturbed_data, t).clone().detach()

        dot = torch.sum(score_minor * score_major, axis=1).cpu()

        cos = torch.nn.CosineSimilarity()(score_minor, score_major).cpu()
        angle = (np.arccos(cos) * 180 / math.pi).cpu().numpy()

        loss_mse = torch.nn.MSELoss()
        loss_cos = torch.nn.CosineSimilarity()

        angle_temp = np.where(angle >= 180, 360-angle, angle)
        weight = torch.ones_like(score_minor) * config.training.lambda_
        weight[angle_temp >= config.training.angle] = 1
        weight = weight.to(config.device)
        loss = loss_mse(input=score_fn_minor(perturbed_data, t), target=weight * score_minor)

        loss.backward()
        optimize_fn(optimizer, minor_model.parameters(), step=state['state']['step'], lr=config.optim.fine_tune_lr)
        state['state']['step'] += 1
        state['state']['ema'].update(minor_model.parameters())

        logging.info(f"epoch: {epoch}, iter: {i}, loss_f:{loss}")


  # test after fine-tuning
  with torch.no_grad():
    eps=1e-05
    
    f1_score_with_N = []
    f1_score_with_Z_major = []

    train_ds_ = transformer.inverse_transform(np.concatenate(train_data_total))
    eval_ds_ = transformer.inverse_transform(eval_ds)

    for i in range(5): # repeat test 5 times 
      synthesized_minor_samples_with_N = []
      synthesized_minor_samples_with_Z_major = []

      for j, state in enumerate(minor_states):
        sde = state['sde']
        model = state['state']['model']
        sampling_fn = state['sampling_fn']

        train_exclude = np.concatenate(train_data_total[np.arange(len(train_data_total))!=j+1])
        sample_size = num_sampling[j]

        perturbed_data = make_noise(config, sde, train_exclude, shape=(sample_size, config.data.image_size), eps=eps,test=True)
        sample1, n = sampling_fn(model, z=perturbed_data) 
        sample2, n = sampling_fn(model, sample_size=sample_size) 

        sample = transformer.inverse_transform(sample1.cpu().detach().numpy())
        synthesized_minor_samples_with_Z_major.append(sample)

        sample = transformer.inverse_transform(sample2.cpu().detach().numpy())
        synthesized_minor_samples_with_N.append(sample)

      synthesized_minor_samples_with_Z_major = np.concatenate(synthesized_minor_samples_with_Z_major)
      synthesized_minor_samples_with_N = np.concatenate(synthesized_minor_samples_with_N)
      print(f"original data: {Counter(train_ds_[:, -1])}")
      logging.info(f"original data: {Counter(train_ds_[:, -1])}")
      print(f"oversampled data: {Counter(np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major])[:, -1])}")
      logging.info(f"oversampled data: {Counter(np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major])[:, -1])}")
      
      scores_with_Z_major = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_Z_major]), meta, config.test.n_iter)
      scores_with_N = evaluation.compute_scores(eval_ds_, np.concatenate([train_ds_, synthesized_minor_samples_with_N]), meta, config.test.n_iter)

      f1_score_with_N.append(scores_with_N['weighted_f1'])
      f1_score_with_Z_major.append(scores_with_Z_major['weighted_f1'])
      
    f1_score_with_N = np.array(f1_score_with_N) 
    f1_score_with_Z_major = np.array(f1_score_with_Z_major) 

    z_major_weighted_f1 = f1_score_with_Z_major.mean(axis=0)
    N_weighted_f1 = f1_score_with_N.mean(axis=0)

    logging.info(f"sampling from z_major: {z_major_weighted_f1}±{f1_score_with_Z_major.std(axis=0)}")
    print(f"z_major: {z_major_weighted_f1}±{f1_score_with_Z_major.std(axis=0)}")
    logging.info(f"sampling from z~N(0, 1): {N_weighted_f1}±{f1_score_with_N.std(axis=0)}")
    print(f"z~N(0, 1): {N_weighted_f1}±{f1_score_with_N.std(axis=0)}")
