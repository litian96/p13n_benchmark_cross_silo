import collections
import functools
import pprint
from pathlib import Path

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import jax_utils
import model_utils
import utils

from trainers.base import BaseTrainerLocal


class IFCA_Benchmark(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    assert self.num_clusters > 0, f'expected at least 1 cluster; got {self.num_clusters}'
    self.ensemble_baseline = args['ifca_ensemble_baseline']
    if self.ensemble_baseline:
      # If running ensemble baseline, always do "warm-starting".
      self.num_warmstart_rounds = self.num_rounds + 1
      print('[IFCA Benchmark] Running FedAvg ensemble baseline')
    else:
      self.num_warmstart_rounds = int((self.num_rounds + 1) * args['ifca_warmstart_frac'])
    print(f'[IFCA Benchmark] Num warm start rounds = {self.num_warmstart_rounds}')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    # Ensure random cluster inits for linear problems (by default they use 0 init).
    if self.args['is_linear']:
      cluster_model_fn = lambda inputs: model_utils.linear_model_fn(inputs, zero_init=False)
    else:
      cluster_model_fn = self.model_fn
    cluster_model = hk.without_apply_rng(hk.transform(cluster_model_fn))
    cluster_params = []
    for k in range(self.num_clusters):
      cluster_params.append(cluster_model.init(key, data_batch))
      key = random.fold_in(key, k)

    params_template = cluster_params[0]
    model_size = jax_utils.num_params(params_template)

    # Optimizer shared for every client (re-init before client work)
    client_opt = optimizers.sgd(self.client_lr)
    # When warm-starting IFCA with FedAvg, we want k independent runs
    warmstart_server_opts = [self.server_opt_fn(params_template=cluster_params[k])
                             for k in range(self.num_clusters)]
    # Keep cluster-specific server optimizers.
    cluster_server_opts = [self.server_opt_fn(params_template=cluster_params[k])
                           for k in range(self.num_clusters)]

    ############################################################################

    def loss_fn(params, batch):
      mean_batch_term = self.data_loss_fn(params, batch)
      flat_params = jax_utils.model_flatten(params)
      l2_term = 0.5 * self.l2_reg * (flat_params @ flat_params)
      return mean_batch_term + l2_term

    @jit
    def batch_update(key, opt_state, batch_idx, batch):
      params = client_opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, batch)
      return key, client_opt.update_fn(batch_idx, mean_grad, opt_state)

    @functools.partial(jit, static_argnums=(1, ))
    def _client_eval_model(params, client_idx):
      return self.data_loss_fn(params, (self.x_train[client_idx], self.y_train[client_idx]))

    def client_select_model(params_list, client_idx):
      params_metrics = np.array([_client_eval_model(params, client_idx) for params in params_list])
      return int(np.argmin(params_metrics))

    def fedavg_round(round_idx, key, selected_clients, global_params, server_opt):
      local_updates = [jax_utils.model_zeros_like(global_params)] * self.num_clients
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
        opt_state = client_opt.init_fn(global_params)
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, opt_state, batch_idx, batch)
        local_updates[t] = jax_utils.model_subtract(client_opt.params_fn(opt_state), global_params)

      # Update global model and reset local updates
      global_params = server_opt.step(global_params, local_updates, self.update_weights)
      return global_params

    ############################################################################

    client_2_cluster, cluster_2_client = None, None
    cluster_sizes = []
    print(f'[IFCA stages] Warm-start / IFCA: {self.num_warmstart_rounds} / {self.num_rounds + 1}')

    # Training loop
    progress_bar = tqdm(range(self.num_rounds + 1),
                        desc=f'[IFCA] {cluster_sizes} Round',
                        disable=(self.args['repeat'] != 1))
    for i in progress_bar:
      key = random.fold_in(key, i)
      warmstart_stage = (i < self.num_warmstart_rounds)
      if warmstart_stage:
        # During warm-start, we run normal FedAvg on the cluster models independently
        for k in range(self.num_clusters):
          key = random.fold_in(key, k)
          selected_clients = list(range(self.num_clients))
          cluster_params[k] = fedavg_round(i,
                                           key,
                                           selected_clients,
                                           cluster_params[k],
                                           server_opt=warmstart_server_opts[k])
      else:
        assert not self.ensemble_baseline
        # Reset cluster membership at every round.
        client_2_cluster = [None] * self.num_clients
        cluster_2_client = [[] for _ in range(self.num_clusters)]
        selected_clients = list(range(self.num_clients))
        local_updates = [jax_utils.model_zeros_like(params_template)] * self.num_clients
        for t in selected_clients:
          key = random.fold_in(key, t)
          # Client creates local dataset
          if self.inner_mode == 'iter':
            batches = (next(self.batch_gen[t]) for _ in range(self.inner_iters))
          else:
            batches = utils.epochs_generator(self.x_train[t],
                                             self.y_train[t],
                                             self.batch_sizes[t],
                                             epochs=self.inner_epochs,
                                             seed=int(key[0]))
          # Client selects cluster; cluster_params should be up-to-date.
          cluster_idx = client_select_model(cluster_params, client_idx=t)
          client_2_cluster[t] = cluster_idx
          cluster_2_client[cluster_idx].append(t)
          chosen_params = cluster_params[cluster_idx]
          # Client trains on selected cluster params.
          opt_state = client_opt.init_fn(chosen_params)
          for batch_idx, batch in enumerate(batches):
            key, opt_state = batch_update(key, opt_state, batch_idx, batch)
          model_update = jax_utils.model_subtract(client_opt.params_fn(opt_state), chosen_params)
          local_updates[t] = model_update

        # Loop through clusters and update based on current assignment.
        for k in range(self.num_clusters):
          if len(cluster_2_client[k]) > 0:  # Skip clusters without clients.
            model_updates = [local_updates[t] for t in cluster_2_client[k]]
            update_weights = [self.update_weights[t] for t in cluster_2_client[k]]
            cluster_params[k] = cluster_server_opts[k].step(cluster_params[k], model_updates,
                                                            update_weights)
        # Write down cluster sizes.
        cluster_sizes = [len(clients) for clients in cluster_2_client]
        progress_bar.set_description(f'[IFCA] {cluster_sizes} Round')
        if not self.args['quiet']:
          print(f'Cluster sizes = {cluster_sizes}', end=' ')
        cluster_size_path = Path(self.args['outdir']) / 'cluster_sizes.txt'
        utils.print_log(cluster_sizes, fpath=cluster_size_path)

      # Evaluation and save logs.
      if i % self.args['eval_every'] == 0:
        if client_2_cluster is None:  # Applies to ensemble benchmark.
          # During warm-starting, let each client select the best shared model.
          selected_indices = [client_select_model(cluster_params, t)
                              for t in range(self.num_clients)]
          local_params = [cluster_params[idx] for idx in selected_indices]
        else:
          local_params = [cluster_params[client_2_cluster[t]] for t in range(self.num_clients)]

        client_metrics = self.eval(local_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics
