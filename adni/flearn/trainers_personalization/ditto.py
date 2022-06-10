import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated
from flearn.utils.model_utils import batch_data, gen_batch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using global-regularized multi-task learning to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.global_model = copy.deepcopy(self.latest_model)
        self.local_models = []
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))
        self.actual_updates = [np.zeros_like(p) for p in self.latest_model]

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        batches = {}

        for idx, c in enumerate(self.clients):
            batches[c] = gen_batch(c.train_data, self.batch_size, (self.num_rounds + 5) * self.local_iters)

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:

                num_test, test_loss_vector = self.test_loss(self.local_models)
                num_train, train_loss_vector = self.train_loss(self.local_models)
                num_val, val_loss_vector = self.validation_loss(self.local_models)

                tqdm.write('At round {} training loss: {}'.format(i, np.mean(train_loss_vector)))
                tqdm.write('At round {} test loss: {}'.format(i, np.mean(test_loss_vector)))
                tqdm.write('individual test loss: {}'.format(test_loss_vector.tolist()))
                tqdm.write('At round {} validation loss: {}'.format(i, np.mean(val_loss_vector)))
                tqdm.write('individual validation loss: {}'.format(val_loss_vector.tolist()))


            csolns = []

            for idx, c in enumerate(self.clients):
                for data_batch in batch_data(c.train_data, self.batch_size):
                    # local
                    self.client_model.set_params(self.local_models[idx])
                    _, grads, _ = c.solve_sgd(data_batch)  

                    for layer in range(len(grads[1])):
                        eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
                        self.local_models[idx][layer] = self.local_models[idx][layer] - self.ft_learning_rate * eff_grad

                # global
                self.client_model.set_params(self.global_model)
                train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)

                updates = [soln[layer] - self.global_model[layer] for layer in range(len(self.global_model))]
                csolns.append((train_samples, updates))

            overall_updates = self.aggregate(csolns)

            self.actual_updates = [0.9 * self.actual_updates[layer] + 0.1 * overall_updates[layer] for layer in
                                   range(len(self.global_model))]
            self.global_model = [self.global_model[layer] + self.server_lr * self.actual_updates[layer] for layer in
                                 range(len(self.global_model))]

