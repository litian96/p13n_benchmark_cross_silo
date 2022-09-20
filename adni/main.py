import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'local', 'clustering', 'ensemble', 'fedavgM',
              'ditto', 'knn']
DATASETS = ['adni']


MODEL_PARAMS = {
    'adni.cnn_regression': (32, )  # image size
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer',
                        type=str,
                        choices=OPTIMIZERS,
                        default='qffedavg')
    parser.add_argument('--dataset',
                        help='name of dataset',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        default='cnn_regression')
    parser.add_argument('--num_rounds',
                        help='number of communication rounds to simulate',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ communication rounds',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per communication round',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size of local training',
                        type=int,
                        default=10)
    parser.add_argument('--local_epochs',
                        help='number of local epochs when clients train on data',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for the inner solver',
                        type=float,
                        default=0.003)
    parser.add_argument('--ft_learning_rate',
                        help='learning rate for finetuning',
                        type=float,
                        default=0.003)
    parser.add_argument('--seed',
                        help='seed for random initialization',
                        type=int,
                        default=0)
    parser.add_argument('--lam',
                        help='lambda in the objective',
                        type=float,
                        default=0.1)
    parser.add_argument('--finetune_epochs',
                        help='finetune for how many epochs',
                        type=int,
                        default=0)
    parser.add_argument('--finetune_iters',
                        help='finetune for how many iterations',
                        type=int,
                        default=0)
    parser.add_argument('--warm_start_iters',
                        help='warm start fedavg rounds for clustering',
                        type=int,
                        default=20)
    parser.add_argument('--num_clusters',
                        help='number of clusters',
                        type=int,
                        default=1)
    parser.add_argument('--local_iters',
                        help='number of local iterations',
                        type=int,
                        default=5)
    parser.add_argument('--server_lr',
                        help='server side learning rate',
                        type=float,
                        default=1.0)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # load selected model
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    if parsed['optimizer'] in ['ditto', 'local', 'clustering', 'ensemble', 'knn']:
        opt_path = 'flearn.trainers_personalization.%s' % parsed['optimizer']
    else:
        opt_path = 'flearn.trainers_global.%s' % parsed['optimizer']

    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    tf.logging.set_verbosity(tf.logging.WARN)

    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    t = optimizer(options, learner, dataset)
    t.train()
    

if __name__ == '__main__':
    main()


