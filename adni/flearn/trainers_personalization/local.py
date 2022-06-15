import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('local')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.local_models = []
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))


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


            for idx, c in enumerate(self.clients):
                self.client_model.set_params(self.local_models[idx])
                soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)
                self.local_models[idx] = copy.deepcopy(soln[1])

