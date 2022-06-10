import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy


from .fedbase import BaseFedarated


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using ensembling')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.fedavg_models = []
        for _ in range(self.num_clusters):
            self.fedavg_models.append(copy.deepcopy(self.latest_model))

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        # first get k federated averaging models

        for model_id in range(self.num_clusters):

            with self.client_model.graph.as_default():
                self.client_model.sess.run(tf.global_variables_initializer())
            self.latest_model = self.client_model.get_params()

            self.actual_updates = [np.zeros_like(p) for p in self.latest_model]

            for i in range(self.num_rounds + 1):
                csolns = []

                for c in self.clients:
                    # communicate the latest model
                    c.set_params(self.latest_model)
                    train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)
                    updates = [soln[layer] - self.latest_model[layer] for layer in range(len(soln))]
                    csolns.append((train_samples, updates))

                overall_updates = self.aggregate(csolns)

                self.actual_updates = [0.9 * self.actual_updates[layer] + 0.1 * overall_updates[layer] for layer in
                                       range(len(self.latest_model))]
                self.latest_model = [self.latest_model[layer] + self.server_lr * self.actual_updates[layer] for layer in
                                     range(len(self.latest_model))]

            self.fedavg_models[model_id] = copy.deepcopy(self.latest_model)

        # then ensemble them by selecting the best model
        best_models = []
        for c in self.clients:
            best_model_id = -1
            loss = 1000000
            for id, model in enumerate(self.fedavg_models):
                self.client_model.set_params(model)
                tmp_loss, _ = c.get_val_loss()  # select the model with the lowest validation loss
                if tmp_loss < loss:
                    loss = tmp_loss
                    best_model_id = id
            best_models.append(best_model_id)

        num_test, test_loss_vector = self.test_clustered(self.fedavg_models, best_models)
        num_train, train_loss_vector = self.train_clustered(self.fedavg_models, best_models)
        num_val, val_loss_vector = self.validation_clustered(self.fedavg_models, best_models)

        tqdm.write('training loss: {}'.format(np.mean(train_loss_vector)))
        tqdm.write('test loss: {}'.format(np.mean(test_loss_vector)))
        tqdm.write('individual test loss: {}'.format(test_loss_vector.tolist()))
        tqdm.write('validation loss: {}'.format(np.mean(val_loss_vector)))
        tqdm.write('individual validation loss: {}'.format(val_loss_vector.tolist()))





