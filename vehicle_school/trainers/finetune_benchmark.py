import functools
from pathlib import Path

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import jax_utils
import utils

from trainers.base import BaseTrainerLocal


class Finetune_Benchmark(BaseTrainerLocal):
  def __init__(self, args, data):
    super().__init__(args, data)
    # Allow more local FT rounds after all FedAvg rounds are finished.
    self.final_ft_epochs = self.num_rounds
    self.args['finetune_lr'] = self.args['finetune_lr'] or self.args['client_lr']
    print('[INFO] Finetune benchmarking')

  def train(self):
    key = random.PRNGKey(self.seed)

    ####### Global: Init params with a batch of data #######
    data_batch = self.x_train[0][:self.batch_size].astype(np.float32)
    global_params = self.model.init(key, data_batch)
    local_updates = [0] * self.num_clients
    local_params = [global_params] * self.num_clients

    # Optimizer shared for every client (re-init before client work)
    opt = optimizers.sgd(self.lr)
    server_opt = self.server_opt_fn(params_template=global_params)

    def loss_fn(params, batch):
      train_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.l2_reg * jax_utils.global_l2_norm_sq(params)
      return train_term + l2_term

    @functools.partial(jit, static_argnums=(1, ))
    def batch_update(key, client_opt, opt_state, batch_idx, batch):
      params = client_opt.params_fn(opt_state)
      key = random.fold_in(key, batch_idx)
      mean_grad = grad(loss_fn)(params, batch)
      return key, client_opt.update_fn(batch_idx, mean_grad, opt_state)

    def finetune(global_params, round_idx, key, epochs):
      """Finetuning starting at a given FedAvg round with checkpoint params."""
      n_clients = self.num_clients
      all_metrics = []
      # Allow each client to search thru a grid of LRs.
      finetune_lrs = self.args['finetune_lrs'] or [self.args['finetune_lr']]
      # Loop through all candidate finetuning LRs for all clients/FT epochs.
      for l, ft_lr in enumerate(finetune_lrs):
        ft_opt = optimizers.sgd(ft_lr)  # New optimizer for every finetuning LR
        ft_params = [global_params] * n_clients  # Start from global checkpoint
        epoch_metrics = []
        for ft_epoch in range(epochs):
          for t in range(n_clients):
            key = random.fold_in(key, t)
            batches = utils.epoch_generator(self.x_train[t],
                                            self.y_train[t],
                                            self.batch_sizes[t],
                                            seed=int(key[0]))
            # Train local params corresponding to current LR.
            ft_opt_state = ft_opt.init_fn(ft_params[t])
            for batch_idx, batch in enumerate(batches):
              key, ft_opt_state = batch_update(key, ft_opt, ft_opt_state, batch_idx, batch)
            ft_params[t] = ft_opt.params_fn(ft_opt_state)

          # Eval after every FT epoch; metrics (K, 3) for LR and Epoch
          client_metrics = self.eval(ft_params, round_idx=f'T{round_idx}_FT{ft_epoch}', save=False)
          epoch_metrics.append(client_metrics)
        all_metrics.append(epoch_metrics)

      ###### Save metrics ######
      # Shape notations: T = epochs, K = num of clients, n_lrs = num of FT LRs to tune.
      all_metrics = np.array(all_metrics)  # Shape (num_lrs, T, K, 3), 3=train/val/test
      val_metrics = all_metrics[..., 1]  # Shape (num_lrs, T, K)
      assert all_metrics.shape == (len(finetune_lrs), epochs, n_clients, 3)
      argrank_fn = np.argmin if self.args['is_regression'] else np.argmax
      rank_fn = np.min if self.args['is_regression'] else np.max
      # Save all metric, in case the following simplification goes wrong.
      outdir = Path(self.args['outdir'])
      np.save(outdir / f't{round_idx}_all_metrics.npy', all_metrics)

      ### Fig. 2(a): Shared FT LR across client
      # (metrics when taking client average first)
      avg_val_metrics = np.mean(val_metrics, axis=2)  # (n_lrs, T)
      best_samelr_idx = argrank_fn(rank_fn(avg_val_metrics, axis=1))  # (scalar)
      # (scalar), rank by validation.
      best_samelr_epoch = argrank_fn(avg_val_metrics[best_samelr_idx])
      best_samelr_metrics = all_metrics[best_samelr_idx, best_samelr_epoch]  # (K, 3).
      utils.print_log(
          np.round(best_samelr_metrics, 6).tolist(),  # (K, 3)
          fpath=outdir / f't{round_idx}_best_samelr_metrics.txt')
      # (metrics when taking client average second)
      best_samelr_idx = argrank_fn(np.mean(rank_fn(val_metrics, axis=1), axis=-1))  # (scalar)
      best_samelr_epochs = argrank_fn(val_metrics[best_samelr_idx], axis=0)  # (K,)
      best_samelr_metrics = all_metrics[best_samelr_idx][best_samelr_epochs,
                                                         np.arange(n_clients)]  # (K, 3)
      utils.print_log(
          np.round(best_samelr_metrics, 6).tolist(),  # (K, 3)
          fpath=outdir / f't{round_idx}_best_samelr_metrics_avg2nd.txt')

      ### Fig. 2(a): Custom FT LR across client
      best_customlr_indices = argrank_fn(rank_fn(val_metrics, axis=1), axis=0)  # (K,)
      # Every client takes its best LR index: (n_lrs, T, K, 3) -> (T, K, 3)
      ft_metrics = []  # (T, K, 3)
      for e in range(epochs):
        epoch_vals = all_metrics[:, e]  # (num_lrs, K, 3)
        epoch_vals = epoch_vals[best_customlr_indices, np.arange(n_clients)]  # (K, 3)
        ft_metrics.append(epoch_vals)
      ft_metrics = np.array(ft_metrics)  # (T, K, 3)
      # Take best point across T by val. Due to LR tuning, each client's best epoch can be different.
      best_customlr_epochs = argrank_fn(ft_metrics[..., 1], axis=0)  # (K,)
      best_customlr_metrics = ft_metrics[best_customlr_epochs, np.arange(n_clients)]  # (K, 3)
      utils.print_log(
          np.round(best_customlr_epochs, 6).tolist(),  # (K,)
          fpath=outdir / f't{round_idx}_best_customlr_epochs.txt')
      utils.print_log(
          np.round(best_customlr_metrics, 6).tolist(),  # (K, 3)
          fpath=outdir / f't{round_idx}_best_customlr_metrics.txt')

      ### Fig. 2(b) / (c) / Table: Clients hurt after FT.
      base_metrics = self.eval(  # (K, 3)
          local_params=[global_params] * n_clients,
          round_idx=f'T{round_idx}_FT_baseline',
          save=False)
      utils.print_log(
          np.round(base_metrics, 6).tolist(),  # (K, 3)
          fpath=outdir / f't{round_idx}_base_metrics.txt')
      base_test = base_metrics[..., 2]  # (K,)
      best_customlr_test = best_customlr_metrics[..., 2]  # (K,)
      if self.args['is_regression']:
        hurt_mask = best_customlr_test > base_test
      else:
        hurt_mask = best_customlr_test < base_test
      num_hurt = np.sum(hurt_mask)  # (scalar)
      clients_hurt = np.where(hurt_mask)[0].tolist()  # (< K, ), list of client indices
      hurt_stats = (num_hurt, n_clients, num_hurt / n_clients, clients_hurt)
      hurt_metrics = (np.round(base_test, 6).tolist(),
                      np.round(best_customlr_test, 6).tolist(),
                      self.train_samples.tolist())
      utils.print_log(hurt_stats, fpath=outdir / f't{round_idx}_hurt_stats.txt')  # (4,)
      utils.print_log(hurt_metrics, fpath=outdir / f't{round_idx}_hurt_metrics.txt')

      # Return the best (with client-specific LR) client metrics after FT
      return best_customlr_metrics  # (K, 3)

    ############################################################################

    # Keep track of best finetuning metrics across different checkpoints.
    best_ft_metrics = None  # (K, 3)
    rounds = tqdm(range(self.num_rounds + 1),
                  desc='[Finetune] Round',
                  disable=(self.args['repeat'] != 1))
    for i in rounds:
      key = random.fold_in(key, i)
      selected_clients = list(range(self.num_clients))
      #### Normal FedAvg training ####
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
        opt_state = opt.init_fn(global_params)
        for batch_idx, batch in enumerate(batches):
          key, opt_state = batch_update(key, opt, opt_state, batch_idx, batch)
        local_updates[t] = jax_utils.model_subtract(opt.params_fn(opt_state), global_params)

      # Update global model and reset local updates
      global_params = server_opt.step(global_params, local_updates, self.update_weights)
      local_updates = [0] * self.num_clients

      # Continue to save FedAvg metrics, on top of FT metrics
      if i % self.args['eval_every'] == 0:
        local_params = [global_params] * self.num_clients
        self.eval(local_params, i, fn_prefix='fedavg_')

      #### Finetune checkpoints ####
      if i % self.args['finetune_every'] == 0:
        ft_epochs = (self.final_ft_epochs if i == self.num_rounds else self.args['finetune_epochs'])
        cur_ft_metrics = finetune(global_params, i, key, epochs=ft_epochs)  # (K, 3)
        if best_ft_metrics is None:
          best_ft_metrics = cur_ft_metrics
        # Compare which is better by validation performance since finetuning from
        # the last FedAvg round may not always be the best.
        cur_val, best_val = np.mean(cur_ft_metrics, axis=0)[1], np.mean(best_ft_metrics, axis=0)[1]
        is_better = (cur_val <= best_val) if self.args['is_regression'] else (cur_val >= best_val)
        if is_better:
          best_ft_metrics = cur_ft_metrics
          outdir = Path(self.args['outdir'])
          utils.print_log(i, fpath=outdir / 'best_ft_fedavg_checkpoint.txt')
          utils.print_log(np.round(best_ft_metrics, 5).tolist(),
                          fpath=outdir / f'client_metrics.txt')

    return best_ft_metrics
