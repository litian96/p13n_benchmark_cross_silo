import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy

from flearn.models.client import Client



class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()
        self.actual_updates = [np.zeros_like(p) for p in self.latest_model]

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:  # there is no user grouping information
            groups = [None for _ in users]

        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self):
        num_samples = []
        tot_correct = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.train_error()
            tot_correct.append(ct*1.0)
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def validation_error(self):
        num_samples = []
        tot_correct = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.validation_error()
            tot_correct.append(ct * 1.0)
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)


    def test_loss(self):
        losses = []
        num_samples = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            cl, ns = c.get_test_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)

    def train_loss(self):
        losses = []
        num_samples = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            cl, ns = c.get_train_loss()
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(losses)


    def validation_loss(self):
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

    def test_resulting_model(self):
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass


    def aggregate(self, wsolns): 
        
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

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
