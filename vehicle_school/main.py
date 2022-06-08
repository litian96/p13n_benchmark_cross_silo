import argparse
import collections
import gc
import multiprocessing as mp
import os
import pprint
import random
import sys
import time
from pathlib import Path

import numpy as np

import utils
import data_utils
import parallel_utils
from trainers.local import Local
from trainers.fedavg import FedAvg
from trainers.ditto import Ditto
from trainers.mocha import Mocha
from trainers.ifca_benchmark import IFCA_Benchmark
from trainers.finetune_benchmark import Finetune_Benchmark


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer',
                        help='the algorithm to run',
                        type=str,
                        choices=('local', 'fedavg', 'finetune', 'ifca',  'mocha', 'ditto'),
                        default='fedavg')
    # Datasets
    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=('vehicle', 'school'),
                        type=str,
                        required=True)
    parser.add_argument('--val_split',
                        help='Validation set split fraction',
                        type=float,
                        default=0.15)
    parser.add_argument('--test_split',
                        help='Test set split fraction',
                        type=float,
                        default=0.15)
    parser.add_argument('--density',
                        type=float,
                        help='Fraction of the local training data to use (for each silo)',
                        default=1.0)
    parser.add_argument('--no_std',
                        help='Disable dataset standardization (vehicle and gleam only)',
                        action='store_true')

    # Learning
    parser.add_argument('-lr', '--client_lr',
                        help='learning rate for client SGD',
                        type=float,
                        default=0.01)
    parser.add_argument('--server_lr',
                        help='learning rate for server opt',
                        type=float,
                        default=1)
    parser.add_argument('--batch_size',
                        help='batch size of inner optimization',
                        type=int,
                        default=32)
    parser.add_argument('-t', '--num_rounds',
                        help='number of communication rounds',
                        type=int,
                        default=6000)
    parser.add_argument('-ee', '--eval_every',
                        help='evaluate every `eval_every` rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round; -1 means use all clients',
                        type=int,
                        default=-1)
    parser.add_argument('--inner_mode',
                        help='How to run inner loop: fixed no. of batches or epochs',
                        type=str,
                        choices=('iter', 'epoch'),
                        default='iter')
    parser.add_argument('--inner_epochs',
                        help='number of epochs per communication round',
                        type=int,
                        default=1)
    parser.add_argument('--inner_iters',
                        help='(deprecated for inner_epochs) number of batches per round',
                        type=int,
                        default=1)
    parser.add_argument('--l2_reg',
                        help='L2 regularization',
                        type=float,
                        default=0.0)
    parser.add_argument('--lam_svm',
                        help='regularization parameter for linear SVM',
                        type=float,
                        default=0.0001)  # this param is kept the same for all methods and for all runs

    # Server optimizers
    parser.add_argument('--server_opt',
                        help='Server optimizer',
                        choices=('fedavg', 'fedavgm', 'fedadam'),
                        type=str,
                        default='fedavg')
    parser.add_argument('--fedavg_momentum',
                        help='momentum for FedAvgM',
                        type=float,
                        default=0.9)
    parser.add_argument('--fedadam_beta1',
                        help='Beta_1 for FedAdam',
                        type=float,
                        default=0.9)
    parser.add_argument('--fedadam_beta2',
                        help='Beta_2 for FedAdam',
                        type=float,
                        default=0.99)
    parser.add_argument('--fedadam_tau',
                        help='Tau for FedAdam (term in denominator)',
                        type=float,
                        default=1e-3)

    # Method specific
    parser.add_argument('--ifca_warmstart_frac',
                        help='Fraction of rounds for running FedAvg as IFCA warm-strating',
                        type=float,
                        default=0.2)
    parser.add_argument('--ifca_ensemble_baseline',
                        help='Run the baseline to ensemble `num_clusters` independent FedAvg models are trained',
                        action='store_true')
    parser.add_argument('--ditto_secret_lr',
                        help='learning rate for updating the secret local model of Ditto',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--lambda',
                        help='parameter for personalization',  # also lambda for MOCHA
                        type=float,
                        default=0.0)
    parser.add_argument('--mocha_outer',
                        help='number of inner rounds to runs per server update',
                        type=int,
                        default=1)
    parser.add_argument('--mocha_mode',
                        help='Primal or dual form of mocha',
                        type=str,
                        choices=('primal', 'dual'),
                        default='primal')
    parser.add_argument('--finetune_every',
                        type=int,
                        help='Integer k such that finetuning is run on every k-round FedAvg checkpoint',
                        default=50)
    parser.add_argument('--finetune_epochs',
                        help='Limit on the number of finetuning epochs',
                        type=int,
                        default=100)  # Default from:
    parser.add_argument('--finetune_lr',
                        help='Finetuning local SGD LR',
                        type=float)
    parser.add_argument('--finetune_lrs',
                        help='sweep client-specific finetuning LR',
                        nargs='+',
                        type=float)
    parser.add_argument('-k', '--num_clusters',
                        help='Number of clusters for FedAvg (IFCA)',
                        type=int,
                        default=1)

    # Hyperparameter parallel sweeping args
    parser.add_argument('--sweep',
                        help='Enable sweeping with the default grid',
                        action='store_true')
    parser.add_argument('--no_per_client_metric',
                        help='Disable logging per-client metric for every round',
                        action='store_true')
    parser.add_argument('--num_procs',
                        help='number of parallel processes for mp.Pool()',
                        type=int,
                        default=os.cpu_count())
    parser.add_argument('--client_lrs',
                        help='Sweep client learning rate',
                        nargs='+',
                        type=float)
    parser.add_argument('--server_lrs',
                        help='Sweep server learning rate',
                        nargs='+',
                        type=float)
    parser.add_argument('--ditto_secret_lrs',
                        help='learning rate for updating the secret local model of Ditto',
                        nargs='+',
                        type=float)
    parser.add_argument('--fedadam_taus',
                        help='Sweep FedAdam adaptivity',
                        nargs='+',
                        type=float)
    parser.add_argument('--lambdas',
                        help='sweep lambda values (regularization strength for MTL)',
                        nargs='+',
                        type=float)
    parser.add_argument('--mocha_outers',
                        help='number of inner rounds to runs per server update',
                        nargs='+',
                        type=int)

    # Training args
    parser.add_argument('--unweighted_updates',
                        help='Disable weighing client model updates by their example counts',
                        action='store_true')

    # Misc args
    parser.add_argument('-o', '--outdir',
                        help=('Directory to store artifacts, under `logs/`.'),
                        type=str)
    parser.add_argument('-r', '--repeat',
                        help=('Number of times to repeat the experiment'),
                        type=int,
                        default=1)
    parser.add_argument('-q', '--quiet',
                        help='Try not to print things',
                        action='store_true')
    parser.add_argument('--no_per_round_log',
                        help='Disable storing eval metrics',
                        action='store_true')

    ############################################################################

    args = parser.parse_args()
    print(f'Command executed: python3 {" ".join(sys.argv)}')

    ####### Args validation #######
    if args.outdir is None:
        print(f'Outdir not provided.', end=' ')
        args.outdir = f'logs/{args.trainer}-{time.strftime("%Y-%m-%d--%H-%M-%S")}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Storing outputs to {args.outdir}')

    if args.seed is None or args.seed < 0:
        print(f'Random seed not provided.', end=' ')
        args.seed = random.randint(0, 2**32 - 1)
    print(f'Using {args.seed} as root seed.')

    ## Finetuning
    if args.trainer == 'finetune' and args.finetune_lr is None:
      print('WARNING: args.finetune_lr not provided, defaulting to client_lr')
      args.finetune_lr = args.client_lr

    ## Problem types
    args.is_regression = (args.dataset in ('school', 'adni'))
    args.is_linear = (args.dataset in ('vehicle', 'school', 'gleam', 'acs_empl'))

    ## Record flags and input command
    args_dict = vars(args)
    with open(os.path.join(args.outdir, 'args.txt'), 'w') as f:
        pprint.pprint(args_dict, stream=f)
    with open(os.path.join(args.outdir, 'command.txt'), 'w') as f:
        print(' '.join(sys.argv), file=f)

    print(f'Args:\n{args_dict}')
    return args_dict


def main(options, run_idx=None):
    options['run_idx'] = run_idx
    # set worker specific config.
    if run_idx is not None:
        options['seed'] += 1000 * run_idx
        options['outdir'] = os.path.join(options['outdir'], f'run{run_idx}')
        os.makedirs(options['outdir'], exist_ok=True)
        print(f'Run {run_idx} uses root seed {options["seed"]}')

    seed = options['seed']
    random.seed(1 + seed)
    np.random.seed(12 + seed)

    ##### Create Datasets #####
    dataset_args = dict(seed=seed,
                        bias=False,
                        val_split=options['val_split'],
                        test_split=options['test_split'],
                        density=options['density'],
                        standardize=(not options['no_std']))
    if options['dataset'] == 'vehicle':
        dataset = data_utils.read_vehicle_data(**dataset_args)
    elif options['dataset'] == 'school':
        dataset = data_utils.read_school_data(**dataset_args)
    else:
        raise ValueError(f'Unknown dataset `{options["dataset"]}`')

    ##### Create Trainers #####
    if options['trainer'] == 'fedavg':
        t = FedAvg(options, dataset)
        result = t.train()
    elif options['trainer'] == 'local':
        t = Local(options, dataset)
        result = t.train()
    elif options['trainer'] == 'ditto':
        t = Ditto(options, dataset)
        result = t.train()
    elif options['trainer'] == 'mocha':
        t = Mocha(options, dataset)
        result = t.train()
    elif options['trainer'] == 'ifca':
        t = IFCA_Benchmark(options, dataset)
        result = t.train()
    elif options['trainer'] == 'finetune':
        t = Finetune_Benchmark(options, dataset)
        result = t.train()
    else:
        raise ValueError(f'Unknown trainer `{options["trainer"]}`')

    # Run garbage collection to ensure finished runs don't keep unnecessary memory
    gc.collect()
    print(f'Outputs stored at {options["outdir"]}')
    return result


if __name__ == '__main__':
    options = read_options()
    print(f'outdir: {options["outdir"]}')

    if options['sweep']:
      # Perform sweep and take stats over repertition
      hparam_names, results = parallel_utils.sweep_grid(main, options)
      avg_results, std_results, full_results = {}, {}, {}

      for hparam_tup, repeat_vals in results.items():  # (n_repeats, K, 3) for repeat_vals
        # Axis=0 to ensure taking stats for train/val/test separately
        repeat_vals = np.array(repeat_vals)
        assert repeat_vals.shape[0] == options['repeat'] and repeat_vals.shape[2] == 3
        avg_clients = np.mean(repeat_vals, axis=1)  # (n_repeats, 3); measures overall perf
        std_clients = np.std(repeat_vals, axis=1)  # (n_repeats, 3); measures fairness
        # Avg/Std results across repeats.
        avg_results[hparam_tup] = np.round([np.mean(avg_clients, axis=0),
                                            np.mean(std_clients, axis=0)], 5).tolist()
        std_results[hparam_tup] = np.round([np.std(avg_clients, axis=0),
                                            np.std(std_clients, axis=0)], 5).tolist()
        full_results[hparam_tup] = [avg_results[hparam_tup], std_results[hparam_tup]]

      # Rank best results by min for regression
      rank_fn = min if options['is_regression'] else max

      # Stats over all sweeps. Take best by mean validation.
      # x[1] = value of dict; x[1][0] = mean result; x[1][0][1] = mean of validation
      val_key, val_res = rank_fn(avg_results.items(), key=lambda x: x[1][0][1])
      test_mean, test_std = val_res[0][2], val_res[1][2]  # val_res=(2, 3)
      std_mean, std_std = std_results[val_key][0][2], std_results[val_key][1][2]
      test_dict = {val_key: [[test_mean, test_std], [std_mean, std_std]]}

      # Save results
      with open(os.path.join(options['outdir'], 'hparams_swept.txt'), 'w') as f:
        pprint.pprint(hparam_names, stream=f)
      with open(os.path.join(options['outdir'], 'full_results.txt'), 'w') as f:
        pprint.pprint(dict(full_results), stream=f)
      with open(os.path.join(options['outdir'], 'best_result.txt'), 'w') as f:
        pprint.pprint(test_dict, stream=f)

      print(f'Swept hparams: {hparam_names}')
      print(f'Test metric w/ best validation perf: {test_dict}')

    # No sweeping
    else:
      results = np.array(parallel_utils.repeat(main, options))  # (n_repeats, K, 3)
      assert results.shape[0] == options['repeat'] and results.shape[2] == 3
      avg_clients = np.mean(results, axis=1)  # (n_repeats, 3); measures overall perf
      std_clients = np.std(results, axis=1)  # (n_repeats, 3); measures fairness
      # Avg/Std results across repeats.
      avg_results = np.round([np.mean(avg_clients, axis=0),
                              np.mean(std_clients, axis=0)], 5).tolist()  # (2, 3)
      std_results = np.round([np.std(avg_clients, axis=0),
                              np.std(std_clients, axis=0)], 5).tolist()  # (2, 3)

      print(f'Final avg results:\n{pprint.pformat(avg_results)}')
      print(f'Final std results:\n{pprint.pformat(std_results)}')

      # [0] -> mean, [1] -> std, [:][2] -> test
      test_mean, test_std = avg_results[0][2], avg_results[1][2]
      std_mean, std_std = std_results[0][2], std_results[1][2]
      print(f'final test metric: {test_mean:.5f} ± {test_std:.5f} ({std_mean, std_std})')

      with open(Path(options['outdir']) / 'final_result.txt', 'w') as f:
          pprint.pprint(results, stream=f)
          print([avg_results, std_results], file=f)

    print(f'Final outputs stored at {options["outdir"]}')
