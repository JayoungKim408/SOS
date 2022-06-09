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

"""Utility functions for computing FID/Inception scores."""

# import jax
import torch
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
from collections import Counter

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = 'assets/stats/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  # num_tpus = jax.local_device_count()
  num_gpus = torch.cuda.device_count()
  # input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  input_tensors = tf.split(input_tensor, num_gpus, axis=0)

  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/TPU:{}' if 'TPU' in str(torch.cuda.device_count()) else '/GPU:{}'
  # device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }


import json
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, silhouette_score, matthews_corrcoef
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"



_MODELS = {
    'binary_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 20
            }
        },
        {
            'class': AdaBoostClassifier,
        },
        {
            'class': LogisticRegression,
            'kwargs': {
                'solver': 'lbfgs',
                'n_jobs': -1,
                'max_iter': 50
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 30,
                'class_weight': 'balanced',
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],
    'regression': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],

    'clustering': [
        {
            'class': KMeans, 
            'kwargs': {
                'n_clusters': 2,
                'n_jobs': -1,
            }
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata, clustering=False): 
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    if clustering:
        model = _MODELS["clustering"]
    else:
        model = _MODELS[metadata['problem_type']]
    return x_train, y_train, x_test, y_test, model


def _evaluate_multi_classification(train, test, metadata):
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        pandas.DataFrame
    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]
        proportion = [  report[i]['support'] / len(y_test) for i in classes]
        weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
                
        f1.append([report[c]['f1-score'] for c in classes] )
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                'weighted_f1': weighted_f1,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return pd.DataFrame(performance)


def _evaluate_binary_classification(train, test, metadata):
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)
    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))

        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)


        acc = accuracy_score(y_test, pred)
        binary_f1 = f1_score(y_test, pred, average='binary')
        macro_f1 = f1_score(y_test, pred, average='macro')
        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]

        f1.append([report[c]['f1-score'] for c in classes] )
        weighted_f1 = report['0']['f1-score'] * report['1']['support']/len(y_test) + report['1']['f1-score'] * report['0']['support']/len(y_test)

        mcc = matthews_corrcoef(y_test, pred)

        precision = precision_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred, average='binary')
        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except:
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "binary_f1": binary_f1,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "matthews_corrcoef": mcc, 
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }
        )
    
    return pd.DataFrame(performance)

def _evaluate_regression(train, test, metadata):
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        explained_variance = explained_variance_score(y_test, pred)
        mean_squared = mean_squared_error(y_test, pred)
        mean_absolute = mean_absolute_error(y_test, pred)



        performance.append(
            {
                "name": model_repr,
                "r2": r2,
                "explained_variance" : explained_variance,
                "mean_squared_error" : mean_squared,
                "mean_absolute_error" : mean_absolute
            }
        )

    return pd.DataFrame(performance)

def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t

def _evaluate_cluster(train, test, metadata):

    x_train, y_train, x_test, y_test, kmeans = _prepare_ml_problem(train, test, metadata, clustering=True)
 

    model_class = kmeans[0]['class']
    model_repr = model_class.__name__
    unique_labels = np.unique(y_train)
    num_columns = metadata['columns'][-1]["size"]
    
    result = []
    for i in range(3):
        model = model_class(n_clusters = num_columns*(i+1))

        if len(unique_labels) == 1:
            result.append([unique_labels[0]] * len(x_test))

        else:
            try:
                model.fit(x_train)
                predicted_label = model.predict(x_test)
            except:
                x_train = x_train.astype(np.float32)
                model.fit(x_train)

                x_test = x_test.astype(np.float32)
                predicted_label = model.predict(x_test)
            try:
                result.append(silhouette_score(x_test, predicted_label, metric='euclidean', sample_size=100))
            except:
                result.append(0)
        

    return pd.DataFrame([{
        "name": model_repr,
        "silhouette_score": np.mean(result),
    }])




_EVALUATORS = {
    'regression': [_evaluate_regression],
    'binary_classification': [_evaluate_binary_classification, _evaluate_cluster],
    'multiclass_classification': [_evaluate_multi_classification, _evaluate_cluster]
}

def compute_scores(test, synthesized_data, metadata, test_iter=5):
    result = pd.DataFrame()

    for evaluator in _EVALUATORS[metadata['problem_type']]:
        scores = pd.DataFrame()
        for i in range(test_iter):

            score = evaluator(synthesized_data, test, metadata) 
            score['test_iter'] = i
            scores = pd.concat([scores, score], ignore_index=True)

        scores = scores.groupby(['test_iter']).mean() 
        result = pd.concat([result, scores], axis=1)

    return result
