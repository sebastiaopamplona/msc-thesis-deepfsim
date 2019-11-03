import random
import math
import pickle
import numpy as np

from utils.constants import WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS
from utils.data.data_generators import WIKI_DataGenerator, WIKI_Uni_Relaxed_DataGenerator

from utils.utils import get_tra_val_tes_size


def ask_for_relaxed_age():
    relaxed_ages = pickle.load(open("{}in\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    while True:
        img = input("Enter the face idx [0, {}]: ".format(len(relaxed_ages)))
        print('{}.png: {}'.format(img, relaxed_ages[int(img)]))


def test_data_generator(data_generator, set_size, batch_size, uni_tra):
    """
    Simulates the generation of the batches.
    """
    num_batches = set_size // batch_size
    print("Expected number of batches: {}".format(num_batches))
    samples_parsed = 0

    for i in range(num_batches):
        bx, _ = data_generator.__getitem__(index=0)
        print('Batch [{}]: {} samples'.format(i, len(bx[0])))
        if uni_tra:
            label_ctr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for label in bx[1]:
                label_ctr[label] = label_ctr[label] + 1

            for label in range(6):
                assert label_ctr[label] == 11

        samples_parsed += len(bx[0])

    assert samples_parsed == num_batches * batch_size
    print('Test passed.')


def test_memory_leak(data_generator, set_size, batch_size):
    num_batches = math.ceil(set_size / batch_size)
    samples_parsed = 0
    for i in range(1, set_size+1):
        concatenated_batches = np.empty(shape=(i, 224, 224, 3))
        print('({}, 224, 224, 3) success'.format(i))
        del concatenated_batches

    # for i in range(num_batches):
    #     bx, _ = data_generator.__getitem__(index=0)
    #     samples_parsed += len(bx[0])
    #     concatenated_batches += bx[0]
    #     print('Batch [{}]: {} samples, {} in total'.format(i, len(bx[0]), samples_parsed))
    #     print(len(concatenated_batches))


def test_different_idxs(testname, idxs1, idxs2):
    for idx1 in idxs1:
        for idx2 in idxs2:
            assert idx1 != idx2

    print("{} OK".format(testname))


def test_WIKI_DataGenerator():
    data_generator_params = {'batch_size': 32,
                             'dim': (224, 224, 3),
                             'embedding_size': 64}

    ages = pickle.load(open('{}ages.pickle'.format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    set_size = len(ages)

    print(set_size)

    tra_size, val_size, tes_size = get_tra_val_tes_size(set_size=set_size,
                                                        split_train_val=90,
                                                        split_train_test=90)

    train_generator = WIKI_DataGenerator(ages=ages[0:tra_size], set_size=tra_size, **data_generator_params)
    validation_generator = WIKI_DataGenerator(ages=ages[tra_size:tra_size + val_size], set_size=val_size,
                                              **data_generator_params)
    test_generator = WIKI_DataGenerator(ages=ages[tra_size + val_size:], set_size=tes_size, **data_generator_params)

    print('Training DataGenerator')
    test_data_generator(data_generator=train_generator,
                        set_size=tra_size,
                        batch_size=32,
                        uni_tra=0)

    print('Validation DataGenerator')
    test_data_generator(data_generator=validation_generator,
                        set_size=val_size,
                        batch_size=32,
                        uni_tra=0)

    print('Testing DataGenerator')
    test_data_generator(data_generator=test_generator,
                        set_size=tes_size,
                        batch_size=32,
                        uni_tra=0)


def test_WIKI_Uni_Relaxed_DataGenerator():
    data_generator_params = {'batch_size': 66,
                             'dim': (160, 160, 3),
                             'embedding_size': 128}

    tra_relaxed_ages = pickle.load(open("{}in\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    tes_relaxed_ages = pickle.load(open("{}out\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    tra_size = len(tra_relaxed_ages)
    tes_size = len(tes_relaxed_ages)

    train_generator = WIKI_Uni_Relaxed_DataGenerator(relaxed_ages=tra_relaxed_ages,
                                                     set_size=tra_size,
                                                     training_flag=1,
                                                     **data_generator_params)

    test_generator = WIKI_Uni_Relaxed_DataGenerator(relaxed_ages=tes_relaxed_ages,
                                                    set_size=tes_size,
                                                    training_flag=0,
                                                    **data_generator_params)

    print('Training WIKI_Uni_Relaxed_DataGenerator')
    test_data_generator(train_generator, tra_size, 66, uni_tra=1)

    print('Testing WIKI_Uni_Relaxed_DataGenerator')
    test_data_generator(test_generator, tes_size, 66, uni_tra=0)


if __name__ == "__main__":
    test_WIKI_Uni_Relaxed_DataGenerator()

