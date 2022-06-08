from collections import Counter
import os
import time

import numpy as np
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def vehicle_stats(data_dir='data/vehicle'):
  mat = scipy.io.loadmat(os.path.join(data_dir, 'vehicle.mat'))
  raw_x, raw_y = mat['X'], mat['Y']  # y in {-1, 1}
  assert len(raw_x) == len(raw_y)
  num_clients = len(raw_x)

  dataset = []
  for i in range(num_clients):
    features, labels = raw_x[i][0], raw_y[i][0].flatten()
    assert len(features) == len(labels)
    counter = Counter(labels)
    print(f'Client {i}:', counter, counter[1] / len(labels))
    dataset.append((features, labels))

  positive_counts = [np.count_nonzero(labels + 1) for feats, labels in dataset]
  positive_percentages = [np.count_nonzero(labels + 1) / len(labels) * 100
                          for feats, labels in dataset]
  print('Vehicle dataset:')
  print('\tnumber of clients:', num_clients, len(raw_y))
  print('\tnumber of examples:', [len(raw_x[i][0]) for i in range(num_clients)])
  print('\tnumber of features:', len(raw_x[0][0][0]))
  print('\tCount of positive labels:', positive_counts)
  print('\tPercentage of positive labels:', np.round(positive_percentages, 2).tolist())


def read_vehicle_data(data_dir='data/vehicle',
                      seed=None,
                      val_split=0.15,
                      test_split=0.15,
                      bias=False,
                      density=1.0,
                      standardize=True):
  """Read Vehicle dataset.

  Args:
    data_dir: directory that stores the `vehicle.mat` file
    seed: random seed for generating the train/test split
    bias: whether to insert a column of 1s to the dataset (after standardizing)
        so that a model bias term is implicitly included.
    density: fraction of the training data on each client to keep; this does not
        affect test examples.
  """
  mat = scipy.io.loadmat(os.path.join(data_dir, 'vehicle.mat'))
  raw_x, raw_y = mat['X'], mat['Y']  # y in {-1, 1}
  assert len(raw_x) == len(raw_y)
  num_clients = len(raw_x)
  print(f'Vehicle: K={num_clients}, seed={seed}, val/test={(val_split, test_split)}, density={density}, std={standardize}')

  dataset = []
  for i in range(num_clients):
    features, label = raw_x[i][0], raw_y[i][0].flatten()
    # Do train/val/test split iteratively
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        label,
                                                        test_size=test_split,
                                                        random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=val_split / (1 - test_split),
                                                      random_state=seed)

    if density < 1:
      num_train_examples = int(density * len(x_train))
      train_mask = sklearn.utils.shuffle(range(len(x_train)), random_state=seed + 1)[:num_train_examples]
      x_train = x_train[train_mask]
      y_train = y_train[train_mask]
    if standardize:
      # Preprocessing using mean/std from training examples, within each silo
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_val = scaler.transform(x_val)
      x_test = scaler.transform(x_test)
    if bias:
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_val = np.c_[x_val, np.ones(len(x_val))]
      x_test = np.c_[x_test, np.ones(len(x_test))]

    # Cast labels {-1, +1} to floats
    dataset.append((x_train, y_train.astype(float),
                    x_val, y_val.astype(float),
                    x_test, y_test.astype(float)))

  x_trains, y_trains, x_vals, y_vals, x_tests, y_tests = zip(*dataset)
  # Since different tasks have differnet data, this is a ragged array
  return (np.array(x_trains, dtype=object), np.array(y_trains, dtype=object),
          np.array(x_vals, dtype=object), np.array(y_vals, dtype=object),
          np.array(x_tests, dtype=object), np.array(y_tests, dtype=object))


def school_stats(data_dir='data/school'):
  mat = scipy.io.loadmat(os.path.join(data_dir, 'school.mat'))
  # Note that the raw data structure is different from vehicles
  raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y is exam score
  assert len(raw_x) == len(raw_y)
  num_clients = len(raw_x)

  print('School dataset:')
  print('number of clients:', num_clients, len(raw_y))
  print('number of examples:', [len(raw_x[i]) for i in range(num_clients)])
  print('number of features:', len(raw_x[0][0]))


def read_school_data(data_dir='data/school',
                     seed=None,
                     val_split=0.15,
                     test_split=0.15,
                     bias=False,
                     standardize=True,
                     **__kwargs):
  """Read School dataset."""
  mat = scipy.io.loadmat(os.path.join(data_dir, 'school.mat'))
  # Note that the raw data structure is different from vehicles
  raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y is exam score
  assert len(raw_x) == len(raw_y)
  num_clients = len(raw_x)

  print(f'School: K={num_clients}, seed={seed}, val/test split={(val_split, test_split)}')
  # Use min/max normalization (1, 70)
  min_y = min([min(raw_y[i].flatten()) for i in range(num_clients)])  # 1
  max_y = max([max(raw_y[i].flatten()) for i in range(num_clients)])  # 70

  dataset = []
  for i in range(num_clients):  # For each client
    features, label = raw_x[i], raw_y[i].flatten()
    # Do train/val/test split iteratively
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        label,
                                                        test_size=test_split,
                                                        random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=val_split / (1 - test_split),
                                                      random_state=seed)
    if standardize:
      # Preprocessing using mean/std from training examples, within each silo
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_val = scaler.transform(x_val)
      x_test = scaler.transform(x_test)
      # For y (scores), use min/max normalization
      y_train = (y_train - min_y) / (max_y - min_y)
      y_val = (y_val - min_y) / (max_y - min_y)
      y_test = (y_test - min_y) / (max_y - min_y)
    if bias:
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_val = np.c_[x_val, np.ones(len(x_val))]
      x_test = np.c_[x_test, np.ones(len(x_test))]

    # features / exam scores should be float
    dataset.append((x_train, y_train.astype(float),
                    x_val, y_val.astype(float),
                    x_test, y_test.astype(float)))

  x_trains, y_trains, x_vals, y_vals, x_tests, y_tests = zip(*dataset)
  # Since different tasks have differnet data, this is a ragged array
  return (np.array(x_trains, dtype=object), np.array(y_trains, dtype=object),
          np.array(x_vals, dtype=object), np.array(y_vals, dtype=object),
          np.array(x_tests, dtype=object), np.array(y_tests, dtype=object))
