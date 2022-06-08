import os
import collections
import itertools
import math
import multiprocessing as mp

import numpy as np


def repeat(main_fn, args: dict):
  num_repeats = args['repeat']
  with mp.Pool(num_repeats + 1) as pool:
    results = [pool.apply_async(main_fn, (args, run_idx))
               for run_idx in range(num_repeats)]
    results = [r.get() for r in results]
    return results  # (n_repeats, K, 3)


def sweep_grid(main_fn, args: dict):
  """Handles repeats, hyperparameter sweeps."""
  out_dir = args['outdir']
  num_repeats = args['repeat']
  # Enumerate all possible hparams to be swept
  all_tags = ['clr', 'slr', 'lam', 'dittolr', 'mochaouter', 'adamtau']
  all_names = ['client_lr', 'server_lr', 'lambda', 'ditto_secret_lr', 'mocha_outer', 'fedadam_tau']
  all_list_names = [name + 's' for name in all_names]
  assert len(all_tags) == len(all_names) == len(all_list_names)

  # Sweep only the specified hparams (proceed even with only one value inside)
  sweep_tuples = []
  for tag, name, list_name in zip(all_tags, all_names, all_list_names):
    if args[list_name] is not None:
      sweep_tuples.append((tag, name, args[list_name]))

  # If no sweep args were specified, default to single-item lists and sweep those.
  if len(sweep_tuples) == 0:
    print('WARNING: len(sweep_tuples) == 0')
    for tag, name in zip(all_tags[:2], all_names[:2]):  # [:2] = just use first 2 tags
      args[list_name] = [args[name]]
      sweep_tuples.append((tag, name, args[list_name]))

  sweep_tags, sweep_names, sweep_values = zip(*sweep_tuples)
  print(f'[INFO] sweep_tags={sweep_tags}, sweep_names={sweep_names}, sweep_values={sweep_values}')

  # Construct hparam grid with cartesian product
  grid = list(itertools.product(*sweep_values))
  grid_str = ','.join([f'{tag}={vals}' for tag, vals in zip(sweep_tags, sweep_values)])

  def runner(pool, hparam_tup):
    # Create "tag1val1_tag2val2_..." string tags
    run_dir =  '_'.join([f'{tag}{val}' for tag, val in zip(sweep_tags, hparam_tup)])
    run_dir = f'{out_dir}/{run_dir}'
    # For update the sweep hparams
    run_args = {**args, **dict(zip(sweep_names, hparam_tup)), 'outdir': run_dir}
    return [pool.apply_async(main_fn, (run_args, run_idx))
            for run_idx in range(num_repeats)]  # (num_repeats, K, 3)

  # Start sweeping
  results = collections.defaultdict(list)
  pool_size = args['num_procs']
  num_pools = math.ceil(len(grid) * num_repeats / args['num_procs'])
  print(f'[INFO] Grid size {len(grid)} * {num_repeats} repeats = {len(grid) * num_repeats} '
        f'w/ ~= {num_pools} pools of size {pool_size}:\n{grid_str}')

  # Reset pool in chunks to prevent memory overload
  grid_pointer, pool_idx = 0, 0
  while grid_pointer < len(grid):
    chunk_size = pool_size // num_repeats
    chunk = grid[grid_pointer:grid_pointer + chunk_size]
    assert len(chunk) * num_repeats <= pool_size, '# parallel procs should be at most the pool size'

    print(f'[INFO] Pool {pool_idx} with {len(chunk) * num_repeats} / {pool_size} procs')
    with mp.Pool(pool_size) as pool:
      # Run in parallel (result is a list of repeats)
      for hparam_tup in chunk:
        results[hparam_tup] = runner(pool, hparam_tup)
      # Collect results
      for hparam_tup in chunk:
        results[hparam_tup] = [r.get() for r in results[hparam_tup]]

    grid_pointer += chunk_size
    pool_idx += 1

  print(f'[INFO] Sweep outputs stored to {out_dir}')
  # Results: dict[hparam_tuple -> (n_repeats, K, 3)]
  return sweep_names, results
