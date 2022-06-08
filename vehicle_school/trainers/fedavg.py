import os
import functools

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import jax_utils
import utils

from trainers.base import BaseTrainerGlobal


class FedAvg(BaseTrainerGlobal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running FedAvg')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)
    server_opt = self.server_opt_fn(params_template=global_params)

    def loss_fn(params, batch):
      train_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term

    @jit
    def batch_update(key, batch_idx, opt_state, batch):
      key = random.fold_in(key, batch_idx)
      params = opt.params_fn(opt_state)
      mean_grad = grad(loss_fn)(params, batch)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[FedAvg] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))
      for t in selected_clients:
        key = random.fold_in(key, t)
        if self.inner_mode == 'iter':
          batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
        else:
          batches = utils.epochs_generator(self.x_train[t],
                                           self.y_train[t],
                                           self.batch_sizes[t],
                                           epochs=self.inner_epochs,
                                           seed=int(key[0]))
        # Local batches
        opt_state = opt.init_fn(global_params)
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, batch_idx, opt_state, batch)
        # Record new model and model diff
        local_updates[t] = jax_utils.model_subtract(opt.params_fn(opt_state), global_params)

      # Update global model and reset local updates
      global_params = server_opt.step(global_params, local_updates, self.update_weights)
      local_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(global_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics