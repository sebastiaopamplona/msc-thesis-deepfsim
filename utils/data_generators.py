import math

import keras
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle

from keras import backend as K

print("w")
print(K.tensorflow_backend._get_available_gpus())

class WIKI_DataGenerator(keras.utils.Sequence):
    def __init__(self, ages, start_idx, set_size, batch_size, dim, embedding_size):
        self.ages = ages
        self.start_idx = start_idx
        self.set_size = set_size
        self.batch_size = batch_size
        self.dim = dim
        self.embedding_size = embedding_size

        # self.dataset_path = '/home/sebastiao/Desktop/PERS/DEV/Datasets/wiki_parsed/already_parsed/mtcnn_extracted/'
        # self.dataset_path = '/home/sebastiao/Desktop/DEV/github/master-thesis/wiki_age/dataset/mtcnn_extracted/'

        self.dataset_path = "C:\\Users\\Sebastião Pamplona\\Desktop\\DEV\\datasets\\mtcnn_extracted\\"

        self.idxs = [i for i in range(start_idx, start_idx + set_size)]
        shuffle(self.idxs)
        self.step = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return math.ceil(self.set_size / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""

        if self.step == self.__len__()-1:
            self.step = 0
            shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """Generates data containing batch_size samples"""

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)

        batch_x = np.zeros((end - start, self.dim[0], self.dim[1], self.dim[2]))
        batch_y = np.zeros(end - start)
        in_batch_idx = 0
        for i in range(start, end):
            idx = self.idxs[i]
            batch_x[in_batch_idx] = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
            batch_y[in_batch_idx] = self.ages[idx]

            in_batch_idx += 1


        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train

        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))
