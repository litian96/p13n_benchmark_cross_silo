import functools

import numpy as np
from tqdm import tqdm

from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers

import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class MRMTL(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running MR-MTL / stateful pFedMe')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)
    server_opt = self.server_opt_fn(params_template=global_params)

    def loss_fn(params, batch, global_params):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(global_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term

    @jit
    def batch_update(key, batch_idx, opt_state, global_params, batch):
      key = random.fold_in(key, batch_idx)
      params = opt.params_fn(opt_state)
      mean_grad = grad(loss_fn)(params, batch, global_params)
      return key, opt.update_fn(batch_idx, mean_grad, opt_state)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[MRMTL] Round',
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
        opt_state = opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, batch_idx, opt_state, global_params, batch)

        # Record new model and model diff
        new_local_params = opt.params_fn(opt_state)
        local_updates[t] = jax_utils.model_subtract(new_local_params, local_params[t])
        local_params[t] = new_local_params

      # Update global model and clear local updates
      global_params = server_opt.step(global_params, local_updates, self.update_weights)
      local_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(local_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics
