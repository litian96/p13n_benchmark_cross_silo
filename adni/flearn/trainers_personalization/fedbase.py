import numpy as np
import tensorflow as tf

from flearn.models.client import Client



class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()  # self.latest_model is the global model

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self, models):
        num_samples = []
        tot_correct = []
        losses = []
        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.train_error()
            tot_correct.append(ct*1.0)
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def validation_error(self, models):
        num_samples = []
        tot_correct = []
        losses = []
        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.validation_error()
            tot_correct.append(ct*1.0)
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test(self, models):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test_loss_global(self):
        losses = []
        num_samples = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            cl, ns = c.get_test_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)

    def train_loss_global(self):
        losses = []
        num_samples = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            cl, ns = c.get_train_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)


    def validation_loss_global(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            cl, ns = c.get_val_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(losses)

    def train_loss(self, models):
        num_samples = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            cl, ns = c.get_train_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(losses)

    def test_loss(self, models):
        num_samples = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            cl, ns = c.get_test_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(losses)

    def validation_loss(self, models):
        num_samples = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            cl, ns = c.get_val_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(losses)

    def test_clustered(self, models, best_model_ids):

        num_samples = []
        losses = []

        for i, c in enumerate(self.clients):
            self.client_model.set_params(models[best_model_ids[i]])
            cl, ns = c.get_test_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)

    def train_clustered(self, models, best_model_ids):

        num_samples = []
        losses = []

        for i, c in enumerate(self.clients):
            self.client_model.set_params(models[best_model_ids[i]])
            cl, ns = c.get_train_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)

    def validation_clustered(self, models, best_model_ids):

        num_samples = []
        losses = []

        for i, c in enumerate(self.clients):
            self.client_model.set_params(models[best_model_ids[i]])

            cl, ns = c.get_val_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)


    def save(self):
        pass


    def aggregate(self, wsolns):

        total_weight = 0.0
        base = [0] * len(wsolns[0][1])

        for (w, soln) in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln


    def simple_average(self, parameters):

        base = [0] * len(parameters[0])

        for p in parameters:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(parameters) for v in base]

        return averaged_params

