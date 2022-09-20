import os
import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data


class Model(object):
    def __init__(self, image_size, optimizer, seed=1):
        self.img_size = image_size

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            #tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.loss, self.predictions, self.embeddings = self.create_model(optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # allow GPU to grow
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())


    def create_model(self, optimizer):
        features = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size], name='features')
        labels = tf.placeholder(tf.float32, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, self.img_size, self.img_size, 1])
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
        preds = tf.layers.dense(dense, units=1)
        loss = tf.reduce_mean(tf.pow(tf.reshape(preds, [-1])-labels, 2))  # MSE error
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return features, labels, train_op, grads, loss, preds, dense

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_embeddings(self, data):
        with self.graph.as_default():
            embeddings = self.sess.run(self.embeddings, feed_dict={self.features: np.array(data['x']),
                                                      self.labels: np.array(data['y'])})
        return embeddings

    def get_predictions(self, data):
        with self.graph.as_default():
            predictions = self.sess.run(self.predictions, feed_dict={self.features: np.array(data['x']),
                                                      self.labels: np.array(data['y'])})
        return predictions

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, mini_batch_data):

        with self.graph.as_default():
            grads = self.sess.run(self.grads,
                                        feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})

        return grads


    def get_loss(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: np.asarray(data['x']), self.labels: np.asarray(data['y'])})
        return loss

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: np.asarray(X), self.labels: np.asarray(y)})
        soln = self.get_params()
        return soln

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op], feed_dict={self.features: mini_batch_data[0], self.labels: mini_batch_data[1]})
        soln = self.get_params()
        return grads, loss, soln

    def solve_iters(self, num_iters, data):
        for i in range(num_iters):
            data_batch = next(data)
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: data_batch[0], self.labels: data_batch[1]})
        soln = self.get_params()
        return soln

    def close(self):
        self.sess.close()

