import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Hyper Cluster to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.clustered_models = []
        self.actual_updates = []
        for _ in range(self.num_clusters):
            self.clustered_models.append(copy.deepcopy(self.latest_model))
            self.actual_updates.append([np.zeros_like(p) for p in self.latest_model])

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        # first get k models by warmstarting with FedAvg

        for model_id in range(self.num_clusters):

            # create an initialization
            with self.client_model.graph.as_default():
                self.client_model.sess.run(tf.global_variables_initializer())
            self.latest_model = self.client_model.get_params()

            for warm_start_iter in range(self.warm_start_iters):
                csolns = []
                for c in self.clients:
                    # communicate the latest model
                    c.set_params(self.latest_model)

                    # solve minimization locally
                    train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)

                    csolns.append((train_samples, soln))

                self.latest_model = self.aggregate(csolns)

            self.clustered_models[model_id] = copy.deepcopy(self.latest_model)


        for i in range(self.num_rounds + 1):

            if i % self.eval_every == 0:
                # first determine which cluster this client belongs to
                best_models = []
                for c in self.clients:
                    best_model_id = -1
                    loss = 1000000
                    for id, model in enumerate(self.clustered_models):
                        self.client_model.set_params(model)
                        tmp_loss, _ = c.get_val_loss()
                        if tmp_loss < loss:
                            loss = tmp_loss
                            best_model_id = id
                    best_models.append(best_model_id)

                num_test, test_loss_vector = self.test_clustered(self.clustered_models, best_models)
                num_train, train_loss_vector = self.train_clustered(self.clustered_models, best_models)
                num_val, val_loss_vector = self.validation_clustered(self.clustered_models, best_models)

                tqdm.write('training loss: {}'.format(np.mean(train_loss_vector)))
                tqdm.write('test loss: {}'.format(np.mean(test_loss_vector)))
                tqdm.write('individual test loss: {}'.format(test_loss_vector.tolist()))
                tqdm.write('validation loss: {}'.format(np.mean(val_loss_vector)))
                tqdm.write('individual validation loss: {}'.format(val_loss_vector.tolist()))


            csolns = []
            for _ in range(self.num_clusters):
                csolns.append([])


            for c in self.clients:
                # select the best model for this client
                best_model_id = -1
                best_loss = 1e6
                for model_id in range(self.num_clusters):
                    c.set_params(self.clustered_models[model_id])
                    val_loss, _ = c.get_val_loss()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_id = model_id
                    elif val_loss == best_loss:  # break the tie uniformly at random
                        best_model_id = np.random.choice([best_model_id, model_id], p=[1-1.0/(model_id+1), 1.0/(model_id+1)])

                if i % 1 == 0:
                    print('round ', i, ' client ', c.id, ' picking model ', best_model_id)

                c.set_params(self.clustered_models[best_model_id])

                train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)

                updates = [soln[layer] - self.clustered_models[best_model_id][layer] for layer in range(len(soln))]

                csolns[best_model_id].append((train_samples, updates))

            # update models
            for model_id in range(self.num_clusters):
                if len(csolns[model_id]) > 0:
                    overall_updates = self.aggregate(csolns[model_id])
                    self.actual_updates[model_id] = [0.9 * self.actual_updates[model_id][layer] + 0.1 * overall_updates[layer] for layer in range(len(overall_updates))]
                    self.clustered_models[model_id] = [self.clustered_models[model_id][layer] + self.server_lr * self.actual_updates[model_id][layer] for layer in range(len(overall_updates))]


