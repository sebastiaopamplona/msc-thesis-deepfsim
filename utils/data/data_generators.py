import keras
import numpy as np

from PIL import Image
from numpy import asarray
from random import shuffle

from utils.constants import WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS

def get_standardized_pixels(filename):
    """
    Standardizes image pixels; Facenet expects standardized pixels as input.

    :param filename: filepath to the image

    :return: standardized pixels
    """
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    return (pixels - mean) / std


class AgeDG(keras.utils.Sequence):
    def __init__(self, dataset_path, ages, set_size,
                 batch_size, embedding_size, img_format, img_dimension):
        assert len(ages) == set_size
        self.dataset_path = dataset_path
        self.ages = ages
        self.set_size = set_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.img_format = img_format
        self.img_dimension = img_dimension
        self.idxs = [i for i in range(0, set_size)]
        shuffle(self.idxs)
        self.step = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return self.set_size // self.batch_size

    def log(self, log_type, msg):
        print("[{}] {}: {}".format(log_type, self.__class__.__name__, msg))

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.step == self.__len__():
            self.step = 0
            self.log(log_type="INFO", msg="shuffling indexes...")
            shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """Generates data containing batch_size samples"""

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)
        batch_x = np.zeros((end - start,
                            self.img_dimension[0],
                            self.img_dimension[1],
                            self.img_dimension[2]))
        batch_y = np.zeros(end - start)
        in_batch_idx = 0
        for i in range(start, end):
            idx = self.idxs[i]
            filename = '{}{}{}'.format(self.dataset_path, idx, self.img_format)
            pixels = get_standardized_pixels(filename)
            batch_x[in_batch_idx] = pixels
            batch_y[in_batch_idx] = self.ages[idx]
            in_batch_idx += 1

        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train
        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))


class AgeIntervalDG(keras.utils.Sequence):
    def __init__(self, dataset_path, age_intervals, set_size, num_i, uni,
                 batch_size, embedding_size, img_format, img_dimension):
        # assert len(age_intervals) == set_size
        self.dataset_path = dataset_path
        self.age_intervals = age_intervals
        self.set_size = set_size
        self.num_i = num_i
        self.uni = uni
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.img_format = img_format
        self.img_dimension = img_dimension
        if uni:
            assert batch_size % num_i == 0
            self.idx_map = {i: age_intervals[i] for i in range(num_i)}
        else:
            self.idxs = [i for i in range(0, set_size)]
        self.step = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return self.set_size // self.batch_size

    def log(self, log_type, msg):
        print("[{}] {}: {}".format(log_type, self.__class__.__name__, msg))

    def __getitem__(self, index):
        """Generate one batch of data"""

        if self.step == self.__len__():
            self.step = 0
            self.log(log_type="INFO", msg="shuffling indexes...")
            if self.uni:
                for i in range(self.num_i):
                    shuffle(self.idx_map[i])
            else:
                shuffle(self.idxs)



        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """Generates data containing batch_size samples"""

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)
        batch_x = np.zeros((end - start,
                            self.img_dimension[0],
                            self.img_dimension[1],
                            self.img_dimension[2]))
        batch_y = np.zeros(end - start)
        in_batch_idx = 0
        if self.uni:
            per_interval = int(self.batch_size / self.num_i)
            for i in range(self.num_i):
                for p in range(per_interval):
                    idx = self.idx_map[i][self.step * per_interval + p]
                    filename = '{}..\\in\\{}.png'.format(self.dataset_path, idx)
                    pixels = get_standardized_pixels(filename)
                    batch_x[in_batch_idx] = pixels
                    batch_y[in_batch_idx] = i
                    in_batch_idx += 1
        else:
            for i in range(start, end):
                idx = self.idxs[i]
                filename = '{}..\\out\\{}.png'.format(self.dataset_path, idx)
                pixels = get_standardized_pixels(filename)
                batch_x[in_batch_idx] = pixels
                batch_y[in_batch_idx] = self.age_intervals[idx]
                in_batch_idx += 1

        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train
        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))


class EigenvaluesDG(keras.utils.Sequence):
    def __init__(self, dataset_path, eigenvalues, set_size,
                 batch_size, embedding_size, img_format, img_dimension):
        assert len(eigenvalues) == set_size
        self.dataset_path = dataset_path
        self.eigenvalues = eigenvalues
        self.set_size = set_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.img_format = img_format
        self.img_dimension = img_dimension
        self.idxs = [i for i in range(0, set_size)]
        shuffle(self.idxs)
        self.step = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return self.set_size // self.batch_size

    def log(self, log_type, msg):
        print("[{}] {}: {}".format(log_type, self.__class__.__name__, msg))

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.step == self.__len__():
            self.step = 0
            self.log(log_type="INFO", msg="shuffling indexes...")
            shuffle(self.idxs)

        batch_x, batch_y = self.__data_generation()
        self.step += 1

        return batch_x, batch_y

    def __data_generation(self):
        """Generates data containing batch_size samples"""

        start = self.step * self.batch_size
        end = min((self.step + 1) * self.batch_size, self.set_size)
        batch_x = np.zeros((end - start,
                            self.img_dimension[0],
                            self.img_dimension[1],
                            self.img_dimension[2]))
        batch_y = np.zeros((end - start, len(self.eigenvalues[0])))
        in_batch_idx = 0
        for i in range(start, end):
            idx = self.idxs[i]
            filename = '{}{}{}'.format(self.dataset_path, idx, self.img_format)
            pixels = get_standardized_pixels(filename)
            batch_x[in_batch_idx] = pixels
            batch_y[in_batch_idx] = self.eigenvalues[idx]
            in_batch_idx += 1

        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train
        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))


# TODO: delete...
class AgeRelaxedIntervalDG(keras.utils.Sequence):
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
        """Denotes the number of batches per epoch"""

        return self.set_size // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""

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
        """Generates data containing batch_size samples."""

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
                    filename = '{}{}.png'.format(self.dataset_path, idx)
                    pixels = get_standardized_pixels(filename)
                    batch_x[in_batch_idx] = pixels
                    batch_y[in_batch_idx] = self.relaxed_ages[idx]
                    in_batch_idx += 1
        else:
            for i in range(start, end):
                idx = self.idxs[i]
                filename = '{}{}.png'.format(self.dataset_path, idx)
                pixels = get_standardized_pixels(filename)
                batch_x[in_batch_idx] = pixels
                batch_y[in_batch_idx] = self.relaxed_ages[idx]

                in_batch_idx += 1


        # to match (from model.fit()): x=[x_train, y_train], y=dummy_train
        return [batch_x, batch_y], np.ones((self.batch_size, self.embedding_size + 1))


