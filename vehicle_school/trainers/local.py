import os

import numpy as np
from tqdm import tqdm

import jax
from jax import random, jit, grad
from jax.example_libraries import optimizers

import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class Local(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Local')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
      key = random.fold_in(key, t)

    # Optimizer shared for every client (re-init before client work); No server optimizer needed.
    opt = optimizers.sgd(self.lr)

    def loss_fn(params, batch):
      train_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term

    @jit
    def batch_update(key, batch_idx, opt_state, batch):
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, batch)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Local] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))
      for t in selected_clients:
        key = random.fold_in(key, t)
        # Batch generator
        if self.inner_mode == 'iter':
          batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          batches = utils.epochs_generator(self.x_train[t],
                                           self.y_train[t],
                                           self.batch_sizes[t],
                                           epochs=self.inner_epochs,
                                           seed=int(key[0]))
        # Local batches
        opt_state = opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, batch_idx, opt_state, batch)

        local_params[t] = opt.params_fn(opt_state)

      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(local_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics
