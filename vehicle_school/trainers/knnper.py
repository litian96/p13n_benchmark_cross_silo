import os
import functools
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import optimizers
import haiku as hk

import jax_utils
import utils

from trainers.base import BaseTrainerGlobal


class kNNPer(BaseTrainerGlobal):
  """Implements kNN-Per (https://arxiv.org/pdf/2111.09360.pdf).

  Note that since we deal with linear models for Vehicle and School datasets,
  we only need to run kNN on the input features directly. The training logic
  is identical to that of FedAvg; we only need to additionally keep local kNN
  models on each client and alter the predictions.
  """
  def __init__(self, args, data):
    super().__init__(args, data)
    print('[INFO] Initializing kNN-Per')
    self.init_knn_models()
    self.init_inference_fns()
    print('[INFO] Running kNN-Per')

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
                  desc='[kNNPer] Round',
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

      # NOTE: since eval does not JIT (we rely on scikit-learn kNN impl), it is
      # a performance bottleneck.
      if i % self.args['eval_every'] == 0:
        client_metrics = self.eval(global_params, i)  # (K, 3)

    # Return metric at final round
    return client_metrics

  ##############################################################################

  def init_knn_models(self):
    """Create kNN models from each client's training set."""
    knn_model_fn = KNeighborsRegressor if self.args['is_regression'] else KNeighborsClassifier
    self.knn_models = []
    for t in range(self.num_clients):
      model = knn_model_fn(n_neighbors=self.args['knn_neighbors'], weights=self.args['knn_weights'])
      model = model.fit(self.x_train[t], self.y_train[t])
      self.knn_models.append(model)

  def init_inference_fns(self):
    """(Re-)initializes the inference functions (loss and prediction) for kNN-Per."""

    @jit
    def linear_model_pred(params, batch_inputs):
      return self.model.apply(params=params, inputs=batch_inputs).squeeze()

    if self.dataset == 'vehicle':

      def knnper_linear_svm_classify(params, batch_inputs, client_idx):
        # Combine SVM prediction with kNN prediction; simply shift P(y = 1 | x)
        # to [-1, 1] for linearly interpolating with the SVM outputs.
        # Assume that the local knn models are initialized at this point.
        knn_model = self.knn_models[client_idx]
        # Eq 6 of arxiv.org/pdf/2111.09360 (default to scikit-learn's impl).
        knn_preds = knn_model.predict_proba(batch_inputs)  # (n, 2) for binary labels
        knn_preds = knn_preds[:, 1] * 2 - 1  # Rescale P(y = 1 | x) to [-1, 1]
        model_preds = linear_model_pred(params, batch_inputs)
        # Eq 7 of arxiv.org/pdf/2111.09360
        preds = self.lam * knn_preds + (1 - self.lam) * model_preds
        return np.sign(preds)

      # NOTE: Cannot JIT because we use scikit kNN models.
      self.knnper_pred_fn = knnper_linear_svm_classify
      # No need to change `data_loss_fn` (hinge loss for Vehicle).
      self.knnper_loss_fn = lambda params, batch, idx: self.data_loss_fn(params, batch)

    elif self.dataset == 'school':

      def knnper_regression_pred(params, batch_inputs, client_idx):
        knn_model = self.knn_models[client_idx]
        knn_preds = knn_model.predict(batch_inputs)
        model_preds = linear_model_pred(params, batch_inputs)
        return self.lam * knn_preds + (1 - self.lam) * model_preds

      def knnper_mse_loss(params, batch, client_idx):
        inputs, targets = batch
        preds = knnper_regression_pred(params, inputs, client_idx)
        per_example_loss = 0.5 * (preds - targets)**2
        return np.mean(per_example_loss)

      # NOTE: Cannot JIT because we use scikit kNN models.
      self.knnper_pred_fn = knnper_regression_pred
      self.knnper_loss_fn = knnper_mse_loss

    else:
      raise ValueError(f'Unsupported dataset: {self.dataset}')

  def eval(self,
           global_params,
           round_idx,
           save=True,
           save_per_client=True,
           fn_prefix='',
           quiet=False):
    """HACK: Custom evaluation function for kNN-Per, as inference uses local datastores.

    The only change is on the calls to `data_loss_fn` and `pred_fn`, which now
    includes the client index for accessing the local kNN models.
    """
    local_params = [global_params] * self.num_clients
    # Compute loss (both regression and classification)
    quiet = quiet or self.args['quiet']
    if self.args['no_per_client_metric']:
      save_per_client = False

    outdir = Path(self.args['outdir'])
    losses = []
    for t in range(self.num_clients):
      # HACK: allow loss_fn to take client index.
      train_loss = self.knnper_loss_fn(local_params[t], (self.x_train[t], self.y_train[t]), t)
      val_loss = self.knnper_loss_fn(local_params[t], (self.x_val[t], self.y_val[t]), t)
      test_loss = self.knnper_loss_fn(local_params[t], (self.x_test[t], self.y_test[t]), t)
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
        # HACK: allow pred_fn to take client index.
        train_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_train[t], t) == self.y_train[t])
        val_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_val[t], t) == self.y_val[t])
        test_acc = np.mean(self.knnper_pred_fn(local_params[t], self.x_test[t], t) == self.y_test[t])
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
