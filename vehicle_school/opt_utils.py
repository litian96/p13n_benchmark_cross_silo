from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map

from jax_utils import model_add
from jax_utils import model_add_scalar
from jax_utils import model_average
from jax_utils import model_multiply
from jax_utils import model_multiply_scalar
from jax_utils import model_divide
from jax_utils import model_sqrt

# NOTE: The server optimizers below can be made functional to improve performance
# by `jit`ing the `step` functions directly.

class FedAvgM:
  def __init__(self, params_template, server_lr, momentum=0.9):
    self.server_lr = server_lr
    self.momentum = momentum
    self.moment_update = tree_map(jnp.zeros_like, params_template)

  def step(self, params, model_updates, update_weights=None):
    average_update = model_average(model_updates, weights=update_weights)
    # Momentum update.
    term_1 = model_multiply_scalar(self.moment_update, self.momentum)
    term_2 = model_multiply_scalar(average_update, 1 - self.momentum)
    self.moment_update = model_add(term_1, term_2)
    final_update = model_multiply_scalar(self.moment_update, self.server_lr)
    return model_add(params, final_update)


class FedAvg(FedAvgM):
  def __init__(self, params_template, server_lr):
    super().__init__(params_template, server_lr, momentum=0.0)


class FedAdam:
  def __init__(self, params_template, server_lr, beta_1=0.9, beta_2=0.99, tau=1e-3):
    self.server_lr = server_lr
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.m_t = tree_map(jnp.zeros_like, params_template)
    self.v_t = tree_map(jnp.zeros_like, params_template)
    self.tau = tau

  def step(self, params, model_updates, update_weights=None):
    average_update = model_average(model_updates, weights=update_weights)
    # Momentum update.
    m_term_1 = model_multiply_scalar(self.m_t, self.beta_1)
    m_term_2 = model_multiply_scalar(average_update, 1 - self.beta_1)
    self.m_t = model_add(m_term_1, m_term_2)
    # Preconditioner update.
    v_term_1 = model_multiply_scalar(self.v_t, self.beta_2)
    sq_update = model_multiply(average_update, average_update)
    v_term_2 = model_multiply_scalar(sq_update, 1 - self.beta_2)
    self.v_t = model_add(v_term_1, v_term_2)
    # Final update term.
    denom = model_add_scalar(model_sqrt(self.v_t), self.tau)
    final_update = model_divide(self.m_t, denom)
    final_update = model_multiply_scalar(final_update, self.server_lr)
    return model_add(params, final_update)
