# Motley: Benchmarking Heterogeneity and Personalization in Federated Learning

This repo contains **cross-silo** experiments for the **Vehicle/School** datasets.

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

### Single run

The general template command for running an experiment is

```bash
bash runners/<dataset>/run_<method>.sh [--flags]
```

For example, local training on Vehicle can be run as

```bash
bash runners/vehicle/run_local.sh --num_rounds 500 --client_lr 0.03 --repeat 5
```

### Hyperparameter sweep

Hyperparameter grid search is done via the `--sweep` flag and at least one list
of hyperparameters to sweep (e.g. `--client_lrs` , `--server_lrs` with a trailing "s";
see `main.py` and `parallel_utils.py` for other options).

For example, we can sweep the client LRs and the server LRs for FedAvg via

```bash
bash runners/vehicle/run_local.sh --num_rounds 500 --repeat 5 \
  --client_lrs "0.01 0.03 0.1" --server_lrs "0.3 1 3" --sweep \
  -o logs/fedavg_sweep
```

The full results will then be stored in `logs/fedavg_sweep/full_result.txt` .

See `experiments/` scripts for examples; e.g. FedAvg training can be done via

```fish
fish experiments/vehicle/fedavg.fish
```
