import pickle
import numpy as np
import os

def load_cifar(path="cifar-10-batches-py"):
    train_batches = []
    train_labels = []

    for i in range(1, 6):
        with open(os.path.join(path, "data_batch_{}".format(i)), 'rb') as file:
            cifar_out = pickle.load(file, encoding='latin1')
            train_batches.append(cifar_out['data'])
            train_labels.extend(cifar_out['labels'])

    X_train = np.vstack(tuple(train_batches)).reshape(-1, 3, 32, 32)
    y_train = np.array(train_labels)

    with open(os.path.join(path, "test_batch"), 'rb') as file:
        cifar_out = pickle.load(file, encoding='latin1')
        X_test = cifar_out['data'].reshape(-1, 3, 32, 32)
        y_test = cifar_out['labels']

    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)
    mean = X_train.mean(axis=(0, 2, 3))
    std = X_train.std(axis=(0, 2, 3))
    X_train = (X_train - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]

    return (X_train, np.array(y_train)), (X_test, np.array(y_test))
