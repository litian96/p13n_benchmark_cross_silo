import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from sklearn.neighbors import NearestNeighbors

from .fedbase import BaseFedarated

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using mean-regularized multi-task learning to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.global_model = copy.deepcopy(self.latest_model)
        self.local_models = []
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:

                num_test, test_loss_vector = self.test_loss_global()
                num_valid, valid_loss_vector = self.validation_loss_global()
                num_train, train_loss_vector = self.train_loss_global()

                tqdm.write('At round {} training loss: {}'.format(i, np.mean(train_loss_vector)))
                tqdm.write('At round {} test loss: {}'.format(i, np.mean(test_loss_vector)))
                tqdm.write('individual test loss: {}'.format(test_loss_vector.tolist()))
                tqdm.write('At round {} validation loss: {}'.format(i, np.mean(valid_loss_vector)))
                tqdm.write('individual validation loss: {}'.format(valid_loss_vector.tolist()))

            csolns = []

            for idx, c in enumerate(self.clients):

                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)

                updates = [soln[layer] - self.latest_model[layer] for layer in range(len(self.latest_model))]

                csolns.append((train_samples, updates))

            overall_updates = self.aggregate(csolns)

            self.actual_updates = [0.9 * self.actual_updates[layer] + 0.1 * overall_updates[layer] for layer in range(len(self.latest_model))]
            self.latest_model = [self.latest_model[layer] + self.server_lr * self.actual_updates[layer] for layer in range(len(self.latest_model))]


        test_errors = []

        for c in self.clients:
            c.set_params(self.latest_model)
            train_embeddings = c.get_embeddings(c.train_data)
            neigh = NearestNeighbors(n_neighbors=10)
            neigh.fit(train_embeddings)

            test_embeddings = c.get_embeddings(c.test_data)
            distances, neighbour_inds = neigh.kneighbors(test_embeddings, 10, return_distance=True)
            global_train_predictions = c.get_predictions(c.train_data).flatten()
            global_test_predictions = c.get_predictions(c.test_data).flatten()
            # linear interpolation between global predictions and local predictions

            print(global_train_predictions.shape)
            local_test_predictions = np.zeros(len(neighbour_inds))
            print(neighbour_inds.shape)
            for j in range(len(neighbour_inds)):
                local_test_predictions[j] = np.mean(global_train_predictions[neighbour_inds[j]])
            final_predictions = self.lam * global_test_predictions + (1-self.lam) * local_test_predictions

            error = np.mean((final_predictions - np.array(c.test_data['y']))**2)
            test_errors.append(error)

        print(test_errors, np.mean(test_errors))





