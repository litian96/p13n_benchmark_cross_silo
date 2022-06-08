import numpy as np
import haiku as hk


def linear_model_fn(inputs, zero_init=True, **kwargs):
  # Zero-initializing linear models is okay and often works better.
  w_init = hk.initializers.Constant(0) if zero_init else None
  return hk.Sequential([
      hk.Flatten(),
      hk.Linear(1, w_init=w_init)
  ])(inputs)
