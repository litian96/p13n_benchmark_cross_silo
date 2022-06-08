import os
import time

import numpy as np
import sklearn


def print_log(message, fpath=None, stdout=False, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)


def gen_batch(data_x, data_y, batch_size, num_iter):
  """Deprecated in favor of `epoch_generator`."""
  index = len(data_y)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data_y)):
      index = 0
      data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=i + 1)

    batched_x = data_x[index:index + batch_size]
    batched_y = data_y[index:index + batch_size]

    yield (batched_x, batched_y)


def epochs_generator(data_x, data_y, batch_size, epochs=1, seed=None):
  for ep in range(epochs):
    gen = epoch_generator(data_x, data_y, batch_size, seed=seed + ep)
    for batch in gen:
      yield batch


def epoch_generator(data_x, data_y, batch_size, seed=None):
  """Generate one epoch of batches."""
  data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=seed)
  # Drop last by default
  epoch_iters = len(data_x) // batch_size
  for i in range(epoch_iters):
    left, right = i * batch_size, (i + 1) * batch_size
    yield (data_x[left:right], data_y[left:right])


def client_selection(seed, total, num_selected, weights=None):
  rng = np.random.default_rng(seed=seed)
  indices = rng.choice(range(total), num_selected, replace=False, p=weights)
  return indices
