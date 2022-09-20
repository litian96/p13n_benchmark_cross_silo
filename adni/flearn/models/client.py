import numpy as np
import random

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, model=None):
        self.model = model
        self.id = id  # integer
        self.group = group

        train_len = len(train_data['x'])
        val_len = int(len(eval_data['x'])/2)

        self.train_data = train_data

        combined = list(zip(eval_data['x'], eval_data['y']))
        random.shuffle(combined)
        eval_data['x'], eval_data['y'] = zip(*combined)

        self.val_data = {}
        self.test_data = {}
        self.val_data['x'], self.val_data['y'] = eval_data['x'][:val_len], eval_data['y'][:val_len]
        self.test_data['x'], self.test_data['y'] = eval_data['x'][val_len:], eval_data['y'][val_len:]

        self.train_samples = len(self.train_data['y'])
        self.val_samples = len(self.val_data['y'])
        self.test_samples = len(self.test_data['y'])


    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, mini_batch_data):
        '''get model gradient'''
        return self.model.get_gradients(mini_batch_data)

    def get_embeddings(self, data):
        return self.model.get_embeddings(data)

    def get_predictions(self, data):
        return self.model.get_predictions(data)

    def get_train_loss(self):
        return self.model.get_loss(self.train_data), self.train_samples

    def get_test_loss(self):
        return self.model.get_loss(self.test_data), self.test_samples

    def get_val_loss(self):
        return self.model.get_loss(self.val_data), self.val_samples

    def get_val_accuracy(self):
        tot_correct, loss = self.model.test(self.val_data)
        return tot_correct * 1.0 / self.val_samples



    def solve_inner(self, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        soln = self.model.solve_inner(self.train_data, num_epochs, batch_size)

        return self.train_samples, soln

    def solve_iters(self, num_iters, batches):

        soln = self.model.solve_iters(num_iters, batches)

        return soln


    def solve_sgd(self, mini_batch_data):
        '''
        run one iteration of mini-batch SGD
        '''
        grads, loss, weights = self.model.solve_sgd(mini_batch_data)
        return (self.train_samples, weights), (self.train_samples, grads), loss


    def train_error(self):

        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.train_samples

    def validation_error(self):

        tot_correct, loss = self.model.test(self.val_data)
        return tot_correct, loss, self.val_samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.test_data)
        return tot_correct, loss, self.test_samples

    def validate(self):
        tot_correct, loss = self.model.test(self.val_data)
        return tot_correct, self.val_samples