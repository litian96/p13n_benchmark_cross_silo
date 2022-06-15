import numpy as np

import tensorflow as tf


def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array

    return client_grads

