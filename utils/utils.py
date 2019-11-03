import numpy
import statistics
import math
import pickle
import random


def get_tra_val_tes_size(set_size, split_train_val, split_train_test):
    """Calculates the sizes of the training, validation and test sets."""
    train_size = int(split_train_test * .01 * set_size)
    test_size = int(set_size - train_size)

    train_size = int(split_train_val * .01 * train_size)
    val_size = set_size - train_size - test_size

    return train_size, val_size, test_size


def get_tra_val_tes_idxs(dataset_size, train_size, val_size):
    idxs = []
    for i in range(dataset_size):
        idxs.append(i)

    idxs = set(idxs)
    train_idxs = random.sample(idxs, train_size)
    for i in train_idxs:
        idxs.remove(i)

    valid_idxs = random.sample(idxs, val_size)
    for i in valid_idxs:
        idxs.remove(i)

    test_idxs = list(idxs)

    return train_idxs, valid_idxs, test_idxs


def to_pickle(obj, filepath):
    pickle_out = open(filepath, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def from_pickle(filepath):
    return pickle.load(open(filepath, 'rb'))


def numpy_calc_mean_and_std(embeddings):
    size = len(embeddings)
    distances = []
    for i in range(size):
        for j in range(i, size):
            print("{}-{}".format(i, j))
            distances.append(numpy.linalg.norm(embeddings[i] - embeddings[j]))

    return statistics.mean(distances), statistics.stdev(distances)


def calc_mean_and_std(embeddings):
    size = len(embeddings)
    distances = []

    for i in range(size):
        for j in range(i, size):
            print("{}-{}".format(i, j))
            distances.append(math.sqrt(sum([(a - b) ** 2 for a, b in zip(embeddings[i], embeddings[j])])))

    return statistics.mean(distances), statistics.stdev(distances)




