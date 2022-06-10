import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy
import json, os


from .fedbase import BaseFedarated
from flearn.utils.model_utils import gen_batch



class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fedavgM to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):

        print('Training with {} workers ---'.format(self.clients_per_round))

        for layer in range(len(self.latest_model)):
            print(self.latest_model[layer].shape)

        batches = {}

        for idx, c in enumerate(self.clients):
            batches[c] = gen_batch(c.train_data, self.batch_size, self.finetune_iters + 10)

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0:
                num_test, test_loss_vector = self.test_loss()  # have set the latest model for all clients
                num_train, train_loss_vector = self.train_loss()
                num_val, val_loss_vector = self.validation_loss()

                tqdm.write('At round {} training loss: {}'.format(i, np.mean(train_loss_vector)))
                tqdm.write('At round {} test loss: {}'.format(i, np.mean(test_loss_vector)))
                tqdm.write('At round {} validation loss: {}'.format(i, np.mean(val_loss_vector)))

            csolns = []

            for c in self.clients:

                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                train_samples, soln = c.solve_inner(num_epochs=self.local_epochs, batch_size=self.batch_size)

                updates = [soln[layer] - self.latest_model[layer] for layer in range(len(self.latest_model))]

                csolns.append((train_samples, updates))

            overall_updates = self.aggregate(csolns)

            self.actual_updates = [0.9 * self.actual_updates[layer] + 0.1 * overall_updates[layer] for layer in range(len(self.latest_model))]
            self.latest_model = [self.latest_model[layer] + self.server_lr * self.actual_updates[layer] for layer in range(len(self.latest_model))]

        out_file_name = 'fedavgM+finetuning_lastlayer_round' + str(i) + '_serverlr'+ str(self.server_lr) + '_clientlr' + str(self.learning_rate) + '_' + str(
            self.ft_learning_rate) + '_seed' + str(self.seed) + '.json'
        output_json = {'training loss': {}, 'test loss': {}, 'validation loss': {}, 'samples': {}}



        for idx, c in enumerate(self.clients):
            c.set_params(self.latest_model)
            tmp_model = copy.deepcopy(self.latest_model)
            output_json['samples'][c.id] = c.test_samples
            output_json['training loss'][c.id] = []
            output_json['test loss'][c.id] = []
            output_json['validation loss'][c.id] = []

            for iter in range(self.finetune_iters):
                c.set_params(tmp_model)
                if iter % 10 == 0:
                    train_loss, train_samples = c.get_train_loss()
                    val_loss, val_samples = c.get_val_loss()
                    test_loss, test_samples = c.get_test_loss()
                    tqdm.write('client {}, {} samples, training loss {}, test loss {}'.format( \
                        idx, c.train_samples, train_loss, test_loss))
                    output_json['training loss'][c.id].append(train_loss.tolist())
                    output_json['test loss'][c.id].append(test_loss.tolist())
                    output_json['validation loss'][c.id].append(val_loss.tolist())
                g = c.get_grads(next(batches[c]))
                for layer in range(len(g)-2, len(g)):
                    tmp_model[layer] = tmp_model[layer] - self.ft_learning_rate * g[layer]

        with open(os.path.join('adni_log/', out_file_name), 'w') as outfile:
            json.dump(output_json, outfile)






