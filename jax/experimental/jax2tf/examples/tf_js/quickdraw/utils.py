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
from jax import numpy as jnp # type: ignore
from jax.experimental import jax2tf # type: ignore
jax2tf.jax2tf.ENABLE_TF_CONVOLUTION = True

import numpy as np # type: ignore
import os
import requests # type: ignore
import tensorflow as tf # type: ignore
from typing import Callable, List, Sequence

class ExportWrapper(tf.train.Checkpoint):

  def __init__(self, fn: Callable, params,
               input_shape: Sequence[int],
               batch_sizes: Sequence[int],
               with_gradient: bool = False):
    """
    Args:
      fn: a function taking two arguments, the parameters and the batch of
        images.
      params: the parameters, as a list/tuple/dictionary of np.ndarray, to be
        used as first argument for `fn`.
     input_shape: the shape of the second argument of `fn` (except the batch size)
     batch_sizes: a sequence of batch sizes for which to save the function.
    """
    super().__init__()
    # Convert fn from JAX to TF.
    self._fn = jax2tf.convert(fn, with_gradient=with_gradient)
    # Create tf.Variables for the parameters.
    self._params = tf.nest.map_structure(
        # If with_gradient=False, we mark the variables behind as non-trainable,
        # or else the Keras model below fails for trying to access them
        # (even with hub.KerasLayer(..., trainable=False), surprisingly).
        lambda param: tf.Variable(param, trainable=with_gradient),
        params)

    # Implement the interface from https://www.tensorflow.org/hub/reusable_saved_models
    self.variables = tf.nest.flatten(self._params)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    for training in (True, False):
      # TODO: batch_size should be None, and nothing else.
      for batch_size in batch_sizes:
        input_spec = tf.TensorSpec([batch_size] + list(input_shape), tf.float32)
        _ = self.__call__.get_concrete_function(input_spec, training=training)
    # If you intend to prescribe regularization terms for users of the model,
    # add them as @tf.functions with no inputs to this list. Else drop this.
    self.regularization_losses: List[Callable] = []

  @tf.function(autograph=False)
  def __call__(self, inputs, training=False):
    del training  # Unused for now.
    # Future directions:
    # - If _fn depends on mode (training or inference), pass on `training`.
    # - If _fn depens on numeric hyperparameters (e.g., dropout rate),
    #   add them as kwargs that have a Python constant as default but
    #   get traced with tf.TensorSpec([], ...).
    # - If _fn needs to execute update ops during training, e.g., to update
    #   batch norm's aggregate stats, make them happen as control dependencies
    #   on outputs if training is true.
    outputs = self._fn(self._params, inputs)
    return outputs

def save_model(fn: Callable, params,
               model_dir: str, *, input_shape: Sequence[int],
               batch_sizes: List[int], with_gradient: bool = False):
  """Saves the SavedModel for a function"""
  wrapper = ExportWrapper(fn, params, input_shape, batch_sizes=batch_sizes,
                          with_gradient=with_gradient)
  # Build signatures
  signatures = {}
  for batch_size in batch_sizes:
    input_spec = tf.TensorSpec([batch_size] + list(input_shape), tf.float32)
    cf = wrapper.__call__.get_concrete_function(input_spec)
    signatures[f"serving_{batch_size}"] = cf
  print(f"Saving the model to {model_dir}")
  tf.saved_model.save(wrapper, model_dir, signatures=signatures)
  return signatures

def download_dataset(dir_path, nb_classes):

  if not os.path.exists(dir_path):
    os.mkdir(dir_path)
  assert os.path.isdir(dir_path), f"{dir_path} exists and is not a directory"

  classes_path = os.path.join(
    os.path.dirname(__file__),
    'third_party/zaidalyafeai.github.io/model2/class_names.txt')
  with open(classes_path, 'r') as classes_file:
    classes = (
      list(map(lambda c: c.strip(), classes_file.readlines()))[:nb_classes])

  url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for cls in classes:
    cls_filename = f"{cls}.npy"
    cls_file_path = os.path.join(dir_path, cls_filename)
    if os.path.exists(cls_file_path):
      print(f'{cls_filename} already exists, skipping')
      continue
    with open(cls_file_path, "wb") as save_file:
      try:
        response = requests.get(url + cls_filename.replace('_', ' '))
        save_file.write(response.content)
        print(f'Successfully fetched {cls_filename}')
      except:
        print(f'Failed to fetch {cls_filename}')
  return classes

def load_classes(dir_path, classes, batch_size=256, test_ratio=0.1,
                 max_items_per_class=4096):
  x, y = np.empty([0, 784]), np.empty([0])
  for idx, cls in enumerate(classes):
    cls_path = os.path.join(dir_path, f"{cls}.npy")
    data = np.load(cls_path)[:max_items_per_class, :]
    labels = np.full(data.shape[0], idx)
    x, y = np.concatenate((x, data), axis=0), np.append(y, labels)

  assert x.shape[0] % batch_size == 0

  x, y = x.astype(jnp.float32) / 255.0, y.astype(jnp.int32)
  # Reshaping to square images
  x = np.reshape(x, (x.shape[0], 28, 28, 1))
  permutation = np.random.permutation(y.shape[0])
  x = x[permutation, :]
  y = y[permutation]

  x = np.reshape(x, [x.shape[0] // batch_size, batch_size] + list(x.shape[1:]))
  y = np.reshape(y, [y.shape[0] // batch_size, batch_size])

  nb_test_elements = int(x.shape[0] * test_ratio)

  x_test, y_test = x[:nb_test_elements], y[:nb_test_elements]
  x_train, y_train = x[nb_test_elements:], y[nb_test_elements:]

  return list(zip(x_train, y_train)), list(zip(x_test, y_test))
