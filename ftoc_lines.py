# Copyright 2018 Google LLC
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

"""A basic MNIST example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""


import time

import numpy.random as npr

import jax
from jax.api import jit, grad, vmap, partial
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import datasets


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def predict(t, params_a, params_b, inputs):
  params = [((1 - t) * wa + t * wb, (1 - t) * ba + t * bb) for
            (wa, ba), (wb, bb) in zip(params_a, params_b)]

  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(activations, w) + b
    activations = jnp.tanh(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params_a, params_b, t, batch):
  inputs, targets = batch
  _predict = partial(predict, params_a=params_a, params_b=params_b, inputs=inputs)
  print('taking grad')
  g = jax.jacobian(_predict)(t)
  print('done')
  return -jnp.mean(jnp.sum(g * targets, axis=1))

def accuracy(params_a, params_b, batch):

  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  _predict = partial(predict, params_a=params_a, params_b=params_b, inputs=inputs)
  print('(test) taking grad')
  g = jax.jacobian(_predict)(t)
  print('done')
  predicted_class = jnp.argmax(g, axis=1)
  return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10
  batch_size = 128

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  @jit
  def update(params_a, params_b, t, batch):
    grad_a, grad_b = grad(loss, (0, 1))(params_a, params_b, t, batch)
    npa = [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params_a, grad_a)]
    npb = [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params_b, grad_b)]
    return npa, npb

  params_a = init_random_params(param_scale, layer_sizes, rng=npr.RandomState(0))
  params_b = init_random_params(param_scale, layer_sizes, rng=npr.RandomState(1))

  for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(num_batches):
      print(f'batch {i}')
      key = jax.random.PRNGKey(i)
      t = jax.random.uniform(key)
      params_a, params_b = update(params_a, params_b, t, next(batches))
    epoch_time = time.time() - start_time

    #train_acc = accuracy(params_a, params_b, (train_images, train_labels))
    test_acc = accuracy(params_a, params_b, (test_images[:10], test_labels[:10]))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    #print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))