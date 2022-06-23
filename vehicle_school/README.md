# Motley: Benchmarking Heterogeneity and Personalization in Federated Learning

This repo contains **cross-silo** experiments for the **Vehicle/School** datasets of the paper "Motley: Benchmarking Heterogeneity and Personalization in Federated Learning". [[PDF](https://arxiv.org/pdf/2206.09262)]

 The implemention is written in Python with [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku).

## Directory Structure

* `trainers/` contains the implementations of the personalization methods
* `runners/` contains helper scripts for starting an experiment with example configs
* `experiments/` contains the main experiment scripts for performing a hyperparameter grid search
* `data_utils.py` contains functions for loading/preprocessing datasets
* `jax_utils.py` contains util functions handling models in JAX
* `model_utils.py` contains model definitions
* `opt_utils.py` contains the implementations of server-side optimizers
* `parallel_utils.py` contains util functions for parallelizing experiments
* `main.py` is the driver script with flags

## Dependencies

* Python dependencies: See `requirements.txt`
* (Optional) [Fish shell](https://fishshell.com/) for `experiment/` scripts (can be re-written in bash)

## Running an experiment

The general template command for running an experiment is

```bash
bash runners/<dataset>/run_<method>.sh [--flags]
```

Below we provide examples for running experiments with a few hyperparameters.
See `main.py` for the full list of hyperparameters.

### Example: single run

Local training on Vehicle can be run as

```bash
bash runners/vehicle/run_local.sh --num_rounds 500 --client_lr 0.03 --repeat 5
```

### Example: hyperparameter sweep

Hyperparameter grid search is done via the `--sweep` flag and at least one list
of hyperparameters to sweep (e.g. `--client_lrs` , `--server_lrs` with a trailing "s";
see `main.py` and `parallel_utils.py` for other options).

For example, we can sweep the client LRs and the server LRs for FedAvg via

```bash
bash runners/vehicle/run_fedavg.sh \
  --num_rounds 500 --repeat 5 \
  --client_lrs "0.01 0.03 0.1" --server_lrs "0.3 1 3" --sweep \
  -o logs/fedavg_sweep
```

The full results will then be stored in `logs/fedavg_sweep/full_results.txt`.
The best test metric over the sweep based on validation performance will be in `logs/fedavg_sweep/best_result.txt`.
The result format is [[mean over clients, std over clients], [std of the mean, std of the std]]; see `main.py` for more details.

See `experiments/` scripts for more examples; e.g. FedAvg training can be done via

```fish
fish experiments/vehicle/fedavg.fish
```

## Hyperparameters

See the appendix of our [paper](https://arxiv.org/pdf/2206.09262) for more details.
