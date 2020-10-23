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
from absl import app # type: ignore
from functools import partial
import os # type: ignore
import time

import jax # type: ignore
from jax import lax
from jax import numpy as jnp
from jax.experimental import jax2tf
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

from jax.config import config # type: ignore
config.config_with_absl()

from quickdraw_lib import categorical_cross_entropy_loss, init_model_and_data, \
                          NB_CLASSES, predict
from utils import download_dataset, ExportWrapper, load_classes

def train_step_jax(optimizer, inputs, labels):
  def loss_fn(params):
    logits = predict(params, inputs)
    return categorical_cross_entropy_loss(logits, labels)
  grad = jax.grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer

def jax_train_one_epoch(optimizer, update_fn, train_ds):
  for inputs, labels in train_ds:
    optimizer = update_fn(optimizer, inputs, labels)
  return optimizer

def tf_train_one_epoch(model, optimizer, loss_fn, train_ds):
  for inputs, labels in train_ds:
    with tf.GradientTape() as tape:
      logits = model(inputs, training=True)
      loss_value = loss_fn(logits, labels)
    #print(model.trainable_variables)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return optimizer

def main(*args):
  dir_path = os.path.join(os.path.dirname(__file__), "data")
  classes = download_dataset(dir_path, NB_CLASSES)

  jax_optimizer, _, jax_ds, _ = init_model_and_data(dir_path, classes)
  tf_ds = list(map(lambda tup:
          (tf.convert_to_tensor(tup[0], dtype=tup[0].dtype),
           tf.convert_to_tensor(tup[1], dtype=tf.int32)), jax_ds))

  tf_model = ExportWrapper(predict, jax_optimizer.target, batch_sizes=[1],
                           input_shape=(28, 28, 1), with_gradient=True)
  
  #save_model(predict, jax_optimizer.target, model_dir, input_shape=(28, 28, 1),
  #             batch_sizes=[1, 256, 256], with_gradient=True))
  #update_fn = partial(train_step, optimizer)

  #jax_update_fn = jax.jit(update_fn)
  #tf_update_fn = tf.function(jax2tf.convert(update_fn), autograph=False,
  #                           experimental_compile=True)

  tf_optimizer = tf.keras.optimizers.Adam()

  tf_loss_fn = (lambda logits, labels:
    tf.keras.losses.categorical_crossentropy(tf.one_hot(labels, NB_CLASSES), 
                                             logits))

  start_time = time.time()
  tf_optimizer = tf_train_one_epoch(tf_model, tf_optimizer, tf_loss_fn, tf_ds)
  epoch_time = time.time() - start_time
  print("TF training took {:0.2f} sec".format(epoch_time))
  start_time = time.time()
  jax_optimizer = jax_train_one_epoch(jax_optimizer, train_step_jax, jax_ds)
  epoch_time = time.time() - start_time
  print("JAX training took {:0.2f} sec".format(epoch_time))

if __name__ == "__main__":
  app.run(main)
