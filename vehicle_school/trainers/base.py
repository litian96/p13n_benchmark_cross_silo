import functools
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from jax import jit
import jax.numpy as jnp
import haiku as hk

import utils
import jax_utils
import model_utils
import opt_utils


class BaseTrainer:
  def __init__(self, args, data):
    x_train, y_train, x_val, y_val, x_test, y_test = data

    self.args = args
    self.seed = args['seed']

    ###### Client #######
    # Check client count consistency
    assert len(x_train) == len(y_train) == len(x_val) == len(y_val) == len(x_test) == len(y_test)
    self.num_clients = len(x_train)
    self.clients_per_round = args['clients_per_round']
    if self.clients_per_round == -1:
      self.clients_per_round = self.num_clients

    ###### Training configs #######
    self.lam = args['lambda']
    self.num_rounds = args['num_rounds']
    self.inner_mode = args['inner_mode']
    self.inner_epochs = args['inner_epochs']
    self.inner_iters = args['inner_iters']
    self.lr = self.client_lr = args['client_lr']
    self.l2_reg = args['l2_reg']
    self.dataset = args['dataset']
    self.train_samples = np.asarray([len(x) for x in x_train])
    self.val_samples = np.asarray([len(x) for x in x_val])
    self.test_samples = np.asarray([len(x) for x in x_test])
    self.num_clusters = args['num_clusters']

    if args['unweighted_updates']:
      self.update_weights = np.ones_like(self.train_samples)
    else:
      self.update_weights = self.train_samples / np.sum(self.train_samples)

    self.batch_size = args['batch_size']
    if self.batch_size == -1:
      # Full batch gradient descent if needed
      self.batch_sizes = [len(x_train[i]) for i in range(self.num_clients)]
    else:
      # Limit batch size to the dataset size, so downstream calculations (e.g. sample rate) don't break
      self.batch_sizes = [min(len(x_train[i]), self.batch_size) for i in range(self.num_clients)]

    ###### Server Optimizer ######
    if not self.args['quiet']:
      print(f'[INFO] Server optimizer: {self.args["server_opt"]}')
    if self.args['server_opt'] == 'fedavgm':
      self.server_opt_fn = functools.partial(opt_utils.FedAvgM,
                                             server_lr=self.args['server_lr'],
                                             momentum=self.args['fedavg_momentum'])
    elif self.args['server_opt'] == 'fedadam':
      self.server_opt_fn = functools.partial(opt_utils.FedAdam,
                                             server_lr=self.args['server_lr'],
                                             beta_1=self.args['fedadam_beta1'],
                                             beta_2=self.args['fedadam_beta2'],
                                             tau=self.args['fedadam_tau'])
    else:  # Fallback to FedAvg
      self.server_opt_fn = functools.partial(opt_utils.FedAvg, server_lr=self.args['server_lr'])

    ###### Learning setup ######
    # Model architecture is fixed for each task.
    if self.dataset == 'vehicle':
      self.data_loss_fn = functools.partial(jax_utils.hinge_loss, reg=self.args['lam_svm'])
      self.pred_fn = jax_utils.linear_svm_classify
      self.model_fn = model_utils.linear_model_fn
    elif self.dataset == 'school':
      self.data_loss_fn = jax_utils.l2_loss
      self.pred_fn = jax_utils.regression_pred
      self.model_fn = model_utils.linear_model_fn
    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

    # Create model architecture & compile prediction/loss function
    self.model = hk.without_apply_rng(hk.transform(self.model_fn))
    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))

    # (DEPRECATED) Batch-wise local data generators; deprecated to use local epochs.
    self.batch_gen = {}
    for i in range(self.num_clients):
      self.batch_gen[i] = utils.gen_batch(x_train[i],
                                          y_train[i],
                                          self.batch_size,
                                          num_iter=(self.num_rounds + 1) * self.inner_iters)

    self.x_train, self.y_train = x_train, y_train
    self.x_val, self.y_val = x_val, y_val
    self.x_test, self.y_test = x_test, y_test

  def train(self):
    raise NotImplementedError(f'BaseTrainer `train()` needs to be implemented')

  def eval(self,
           local_params,
           round_idx,
           save=True,
           save_per_client=True,
           fn_prefix='',
           quiet=False):
    # Compute loss (both regression and classification)
    quiet = quiet or self.args['quiet']
    if self.args['no_per_client_metric']:
      save_per_client = False

    outdir = Path(self.args['outdir'])
    losses = []
    for t in range(self.num_clients):
      train_loss = self.data_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]))
      val_loss = self.data_loss_fn(local_params[t], (self.x_val[t], self.y_val[t]))
      test_loss = self.data_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]))
      losses.append((train_loss, val_loss, test_loss))

    losses = np.array(losses).astype(float)  # (K, 3); float64 for rounding str
    # Unweighted averaging of client metrics
    avg_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)  # (3,)

    if self.args['is_regression']:
      metrics, avg_metric, std_metric = losses, avg_loss, std_loss
    else:
      # Classification additionally computes accuracy
      accs = []
      for t in range(self.num_clients):
        train_acc = jnp.mean(self.pred_fn(local_params[t], self.x_train[t]) == self.y_train[t])
        val_acc = jnp.mean(self.pred_fn(local_params[t], self.x_val[t]) == self.y_val[t])
        test_acc = jnp.mean(self.pred_fn(local_params[t], self.x_test[t]) == self.y_test[t])
        accs.append((train_acc, val_acc, test_acc))

      metrics = accs = np.array(accs).astype(float)  # (K, 3)
      avg_metric = avg_acc = np.mean(accs, axis=0)  # (3,)
      std_metric = std_acc = np.std(accs, axis=0)  # (3,)
      # Also save loss when doing classification
      if save:
        if save_per_client:
          utils.print_log(np.round(losses, 5).tolist(),
                          fpath=outdir / f'{fn_prefix}client_losses.txt')
        utils.print_log(np.round(avg_loss, 5).tolist(), fpath=outdir / f'{fn_prefix}avg_losses.txt')
        utils.print_log(np.round(std_loss, 5).tolist(), fpath=outdir / f'{fn_prefix}std_losses.txt')

    # Save avg / per-client metrics (both classification and regression)
    assert metrics.shape == (self.num_clients, 3)
    if save:
      if save_per_client:
        utils.print_log(np.round(metrics, 5).tolist(),
                        fpath=outdir / f'{fn_prefix}client_metrics.txt')
      utils.print_log(np.round(avg_metric, 5).tolist(),
                      fpath=outdir / f'{fn_prefix}avg_metrics.txt')
      utils.print_log(np.round(std_metric, 5).tolist(),
                      fpath=outdir / f'{fn_prefix}std_metrics.txt')

    if not quiet:
      print(f'Round {round_idx}, avg metric train/val/test: {np.round(avg_metric, 5)}')

    return metrics  # (K, 3)


class BaseTrainerLocal(BaseTrainer):
  def __init__(self, args, data):
    super().__init__(args, data)


class BaseTrainerGlobal(BaseTrainer):
  def __init__(self, params, data):
    super().__init__(params, data)

  def eval(self, params, round_idx):
    local_params = [params] * self.num_clients
    return super().eval(local_params, round_idx)
