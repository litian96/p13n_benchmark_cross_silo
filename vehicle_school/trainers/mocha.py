from typing import List

import numpy as np
from tqdm import tqdm

import haiku as hk
import jax.numpy as jnp
from jax import random, jit, grad
from jax.example_libraries import optimizers

import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class Mocha(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Mocha Primal')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_updates = [0] * self.num_clients
    Sigma = np.eye(self.num_clients) * (1.0 / self.num_clients)

    # Optimizer shared for every client (re-init before client work)
    # Note that adaptive server optimizers don't apply due to the fixed update rule.
    opt = optimizers.sgd(self.lr)

    @jit
    def stack_params(local_params: List[hk.Params]):
      return jnp.array([jax_utils.model_flatten(p) for p in local_params])

    @jit
    def update_sigma(local_params: List[hk.Params]):
      epsil = 1e-8
      params_mat = stack_params(local_params)  # (n, d)
      A = params_mat @ params_mat.T
      D, V = jnp.linalg.eigh(A)
      D = (D * (D > epsil)) + epsil * (D <= epsil)
      sqm = jnp.sqrt(D)
      sqm = sqm / jnp.sum(sqm)
      Sigma = V @ jnp.diag(sqm) @ V.T
      return Sigma

    def loss_fn(params, params_mat, client_idx, Sigma, batch):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      # Ensure current param included for autodiff
      params_mat = params_mat.at[client_idx].set(flat_params)
      reg_term = 0.5 * self.lam * jnp.trace(params_mat.T @ Sigma @ params_mat)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + reg_term + l2_term

    @jit
    def batch_update(key, batch_idx, opt_state, params_mat, client_idx, Sigma, batch):
      params = opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, params_mat, client_idx, Sigma, batch)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Mocha] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      # Mocha outer loop
      for j in range(self.args['mocha_outer']):
        key = random.fold_in(key, j)
        selected_clients = list(range(self.num_clients))
        # Jax/Haiku is functional; so `local_params` can be accessed
        # by all clients simultaneously as long as we dont overwrite it.
        new_local_params = [None] * self.num_clients

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
          params_mat = stack_params(local_params)
          opt_state = opt.init_fn(local_params[t])
          for batch_idx, batch in enumerate(batches):
            key, opt_state = batch_update(key, batch_idx, opt_state, params_mat, t, Sigma, batch)

          # Record new model and model diff
          new_local_params[t] = opt.params_fn(opt_state)

        # After every Mocha outer iteration, updates all weights together
        local_params = new_local_params

      # Update Sigma after `mocha_outer` iterations of simultaneous client updates.
      Sigma = update_sigma(local_params)

      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(local_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics
