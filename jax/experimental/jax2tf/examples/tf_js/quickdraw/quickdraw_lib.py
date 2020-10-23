# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os # type: ignore
import time
from typing import Callable

import flax # type: ignore
from flax import linen as nn
from flax.training import common_utils # type: ignore

import jax # type: ignore
from jax import lax
from jax import numpy as jnp
import numpy as np # type: ignore
from tensorflowjs.converters import convert_tf_saved_model # type: ignore

from utils import download_dataset, load_classes, save_model

NB_CLASSES = 100

# The code below is an adaptation for Flax from the work published here:
# https://blog.tensorflow.org/2018/07/train-model-in-tfkeras-with-colab-and-run-in-browser-tensorflowjs.html

class QuickDrawModule(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1)) # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)

    x = nn.Dense(features=NB_CLASSES)(x)
    x = nn.softmax(x)

    return x

def predict(params, inputs):
  """A functional interface to the trained Module."""
  return QuickDrawModule().apply({'params': params}, inputs)

def categorical_cross_entropy_loss(logits, labels):
  onehot_labels = common_utils.onehot(labels, logits.shape[-1])
  return jnp.mean(-jnp.sum(onehot_labels * jnp.log(logits), axis=1))

def update(optimizer, inputs, labels):
  def loss_fn(params):
    logits = predict(params, inputs)
    return categorical_cross_entropy_loss(logits, labels)
  grad = jax.grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer

def accuracy(predict: Callable, params, dataset):
  def top_k_classes(x, k):
    bcast_idxs = jnp.broadcast_to(np.arange(x.shape[-1]), x.shape)
    sorted_vals, sorted_idxs = lax.sort_key_val(x, bcast_idxs)
    topk_idxs = (
        lax.slice_in_dim(sorted_idxs, -k, sorted_idxs.shape[-1], axis=-1))
    return topk_idxs
  def _per_batch(inputs, labels):
    logits = predict(params, inputs)
    predicted_classes = top_k_classes(logits, 1)
    predicted_classes = predicted_classes.reshape((predicted_classes.shape[0],))
    return jnp.mean(predicted_classes == labels)
  batched = [_per_batch(inputs, labels) for inputs, labels in dataset]
  return jnp.mean(jnp.stack(batched))

def train_one_epoch(optimizer, train_ds):
  for inputs, labels in train_ds:
    optimizer = jax.jit(update)(optimizer, inputs, labels)
  return optimizer

def init_model_and_data(dir_path, classes):
  train_ds, test_ds = load_classes(dir_path, classes)
  rng = jax.random.PRNGKey(0)
  init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = QuickDrawModule().init(rng, init_shape)["params"]
  optimizer = flax.optim.Adam(
      learning_rate=0.001, beta1=0.9, beta2=0.999).create(initial_params)
  return optimizer, initial_params, train_ds, test_ds

def train(dir_path, classes):
  optimizer, params, train_ds, test_ds = init_model_and_data(dir_path, classes)
  for epoch in range(5):
    start_time = time.time()
    optimizer = train_one_epoch(optimizer, train_ds)
    epoch_time = time.time() - start_time
    train_acc = accuracy(predict, optimizer.target, train_ds)
    test_acc = accuracy(predict, optimizer.target, test_ds)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

  return optimizer.target
