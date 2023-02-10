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
### Some libraries are not imported: jax, tensorflow, and tensorflow_datasets.
### New modules are imported: models.tabular_utils, and datasets_tabular.
### Some functions are removed:
## crop_resize, resize_small, and central_crop. 
### A function, get_dataset() is newly defined based on tabular datasets.
###########################################################################


"""Return training and evaluation/test datasets from config files."""

import torch
import numpy as np
from models.tabular_utils import GeneralTransformer
from datasets_tabular import load_data


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size


  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({torch.cuda.device_count()})')


  train, test, cols = load_data(config.data.dataset)
  labels = train.groupby(train.iloc[:, -1]).size()
  
  major = []
  minor = []
  train_major = []
  train_minor = []
  
  for i, val in enumerate(list(labels >= labels.max())):

    if val:
      train_major.append(np.array(train[train.iloc[:, -1] == i]))
      major.append(i)
    else:
      train_minor.append(np.array(train[train.iloc[:, -1] == i]))
      minor.append(i)

  transformer = GeneralTransformer()
  data = np.concatenate([train, test])
  transformer.fit(data, cols[0], cols[1])

  train_major_lst = [ transformer.transform(data) for data in train_major ] 
  train_minor_lst = [ transformer.transform(data) for data in train_minor ] 
  test = transformer.transform(np.array(test))

  return np.array(train_major_lst + train_minor_lst), test, (transformer, cols[2]), major, minor, labels
