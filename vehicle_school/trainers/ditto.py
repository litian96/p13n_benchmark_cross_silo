import functools

import numpy as np
from tqdm import tqdm

from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers

import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class Ditto(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Running Ditto')

    # Since Ditto take multiple inner iters, we overwrite data generator
    self.global_iters = 1
    self.local_iters = 1
    step_factor = self.global_iters + self.local_iters

    for t in range(self.num_clients):
      gen_batch_iters = step_factor * (self.num_rounds + 1) * self.inner_iters
      self.batch_gen[t] = utils.gen_batch(self.x_train[t],
                                          self.y_train[t],
                                          self.batch_sizes[t],
                                          num_iter=gen_batch_iters)

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_params = []
    for t in range(self.num_clients):
      local_params.append(self.model.init(key, data_batch))
    local_global_updates = [0] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    global_client_opt = optimizers.sgd(self.lr)
    secret_client_opt = optimizers.sgd(self.args['ditto_secret_lr'])
    server_opt = self.server_opt_fn(params_template=global_params)

    def loss_fn(params, batch, global_params):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      model_diff = flat_params - jax_utils.model_flatten(global_params)
      prox_term = 0.5 * self.lam * (model_diff @ model_diff)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + prox_term + l2_term

    @functools.partial(jit, static_argnums=(0, ))
    def batch_update(client_opt, key, batch_idx, opt_state, global_params, batch):
      key = random.fold_in(key, batch_idx)
      params = client_opt.params_fn(opt_state)
      mean_grad = grad(loss_fn)(params, batch, global_params)
      return key, client_opt.update_fn(batch_idx, mean_grad, opt_state)

    ############################################################################

    # Training loop
    for i in tqdm(range(self.num_rounds + 1),
                  desc='[Ditto] Round',
                  disable=(self.args['repeat'] != 1)):
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))
      for t in selected_clients:
        key = random.fold_in(key, t)
        # Batch generator
        if self.inner_mode == 'iter':
          global_batches = (next(self.batch_gen[t])
                            for _ in range(self.inner_iters * self.global_iters))
          local_batches = (next(self.batch_gen[t])
                           for _ in range(self.inner_iters * self.local_iters))
        else:
          epoch_gen_fn = functools.partial(utils.epochs_generator, self.x_train[t], self.y_train[t],
                                           self.batch_sizes[t])
          global_batches = epoch_gen_fn(epochs=self.inner_epochs * self.global_iters,
                                        seed=int(key[0]))
          local_batches = epoch_gen_fn(epochs=self.inner_epochs * self.local_iters,
                                       seed=int(key[1]))

        # Global model updates
        opt_state = global_client_opt.init_fn(global_params)
        for batch_idx, batch in enumerate(global_batches):
          # We want prox term to be 0 for global update
          prox_params = global_client_opt.params_fn(opt_state)
          key, opt_state = batch_update(global_client_opt, key, batch_idx, opt_state, prox_params,
                                        batch)
        new_global_params = global_client_opt.params_fn(opt_state)

        # Local (secrete) model updates
        opt_state = secret_client_opt.init_fn(local_params[t])
        for batch_idx, batch in enumerate(local_batches):
          prox_params = global_params
          key, opt_state = batch_update(secret_client_opt, key, batch_idx, opt_state, prox_params,
                                        batch)
        new_local_params = secret_client_opt.params_fn(opt_state)

        # Record new *local* model and *global* model diff
        local_global_updates[t] = jax_utils.model_subtract(new_global_params, global_params)
        local_params[t] = new_local_params

      # Update global model and clear local updates
      global_params = server_opt.step(global_params, local_global_updates, self.update_weights)
      local_global_updates = [0] * self.num_clients

      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(local_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics
