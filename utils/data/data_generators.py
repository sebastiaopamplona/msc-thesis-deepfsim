import random
import keras
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle

from utils.constants import WIKI_ALIGNED_MTCNN_160_ABS, WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS


def age_to_relaxed_interval(age):
    """
       0: [18, 21]
       1: [22, 26]
       2: [27, 32]
       3: [33, 38]
       4: [39, 45]
       5: [46, 52]
    """

    if 18 <= age <= 21:
        return 0
    elif 22 <= age <= 26:
        return 1
    elif 27 <= age <= 32:
        return 2
    elif 33 <= age <= 38:
        return 3
    elif 39 <= age <= 45:
        return 4
    elif 46 <= age <= 52:
        return 5

    return -1


class WIKI_DataGenerator(keras.utils.Sequence):
    def __init__(self, ages, set_size, batch_size, dim, embedding_size):
        assert len(ages) == set_size

        self.ages = ages

        distr = {}
        for age in ages:
            try:
                ctr = distr[age]
                distr[age] = ctr + 1
            except Exception as e:
                distr[age] = 1

        ks = list(distr.keys())
        for k in ks:
            distr[k] = round(distr[k] / len(ages) * 100, 2)

        print(distr)

        # distr = {}
        # for age in ages:
        #     try:
        #         ctr = distr[age]
        #         distr[age] = ctr + 1
        #     except Exception as e:
        #         distr[age] = 1
        #
        # print(distr)
        # print(">>>>>>>>>>>>>>>>>>>>>   {}".format(set_size))

        self.set_size = set_size
        self.batch_size = batch_size
        self.dim = dim
        self.embedding_size = embedding_size

        self.dataset_path = WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS

        self.idxs = [i for i in range(0, set_size)]

        shuffle(self.idxs)
        self.step = 0

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """

        return self.set_size // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        if self.step == self.__len__() - 1:
            self.step = 0
            print("**********SHUFFLING**********")
            shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """
        Generates data containing batch_size samples
        """

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


class COPYOF_WIKI_Uni_Relaxed_DataGenerator(keras.utils.Sequence):
    def __init__(self, relaxed_ages, set_size, batch_size, dim, embedding_size, training_flag):
        assert len(relaxed_ages) == set_size
        assert batch_size == 66

        self.relaxed_ages = relaxed_ages
        self.set_size = set_size
        self.batch_size = batch_size
        self.dim = dim
        self.embedding_size = embedding_size
        self.training_flag = training_flag
        if training_flag:
            self.dataset_path = "{}in\\".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS)
            self.idx_map = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for i in range(len(relaxed_ages)):
                self.idx_map[relaxed_ages[i]].append(i)

            for i in range(6):
                assert len(self.idx_map[i]) == 3498
        else:
            self.dataset_path = "{}out\\".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS)
            self.idxs = [i for i in range(0, set_size)]
            shuffle(self.idxs)

        self.step = 0

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """

        return self.set_size // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        if self.step == self.__len__() - 1:
            self.step = 0
            print("**********SHUFFLING**********")
            if self.training_flag:
                for i in range(6):
                    print(self.idx_map[i])
                    shuffle(self.idx_map[i])
                    print(self.idx_map[i])
            else:
                shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """
        Generates data containing batch_size samples
        """

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)

        batch_x = np.zeros((end - start, self.dim[0], self.dim[1], self.dim[2]))
        batch_y = np.zeros(end - start)
        in_batch_idx = 0
        if self.training_flag:
            # 6 classes
            for i in range(6):
                # 1 of each class (batch_size always 66, for testing purposes)
                for k in range(11):
                    idx = self.idx_map[i][self.step * 11 + k]
                    batch_x[in_batch_idx] = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
                    batch_y[in_batch_idx] = self.relaxed_ages[idx]

                    in_batch_idx += 1
        else:
            for i in range(start, end):
                idx = self.idxs[i]
                batch_x[in_batch_idx] = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
                batch_y[in_batch_idx] = self.relaxed_ages[idx]

                in_batch_idx += 1


        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train

        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))


class WIKI_Uni_Relaxed_DataGenerator(keras.utils.Sequence):
    def __init__(self, relaxed_ages, set_size, batch_size, dim, embedding_size, training_flag):
        assert len(relaxed_ages) == set_size
        assert batch_size == 66

        self.relaxed_ages = relaxed_ages
        self.set_size = set_size
        self.batch_size = batch_size
        self.dim = dim
        self.embedding_size = embedding_size
        self.training_flag = training_flag
        if training_flag:
            self.dataset_path = "{}in\\".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS)
            self.idx_map = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for i in range(len(relaxed_ages)):
                self.idx_map[relaxed_ages[i]].append(i)

            for i in range(6):
                assert len(self.idx_map[i]) == 3498
        else:
            self.dataset_path = "{}out\\".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS)
            self.idxs = [i for i in range(0, set_size)]
            shuffle(self.idxs)

        self.step = 0

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """

        return self.set_size // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        if self.step == self.__len__() - 1:
            self.step = 0
            print("**********SHUFFLING**********")
            if self.training_flag:
                for i in range(6):
                    print(self.idx_map[i])
                    shuffle(self.idx_map[i])
                    print(self.idx_map[i])
            else:
                shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """
        Generates data containing batch_size samples
        """

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)

        batch_x = np.zeros((end - start, self.dim[0], self.dim[1], self.dim[2]))
        batch_y = np.zeros(end - start)
        in_batch_idx = 0
        if self.training_flag:
            # 6 classes
            for i in range(6):
                # 1 of each class (batch_size always 66, for testing purposes)
                for k in range(11):
                    idx = self.idx_map[i][self.step * 11 + k]
                    batch_x[in_batch_idx] = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
                    batch_y[in_batch_idx] = self.relaxed_ages[idx]

                    in_batch_idx += 1
        else:
            for i in range(start, end):
                idx = self.idxs[i]
                image = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
                print(image)
                image /= 255.
                batch_x[in_batch_idx] = plt.imread('{}{}.png'.format(self.dataset_path, idx))[:, :, :3]
                batch_y[in_batch_idx] = self.relaxed_ages[idx]

                in_batch_idx += 1


        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train

        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))