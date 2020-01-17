import random
import math
import pickle
import imgaug as ia
import numpy as np

from utils.constants import WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS, \
    IMDB_ALIGNED_MTCNN_160_ABS, WIKI_18_58_160, \
    WIKI_18_58_224, WIKI_ALIGNED_UNI_160, WIKI_AUGMENTED_UNI_160, IMDB_ALIGNED
from utils.data.data_generators import AgeDG, EigenvectorsDG, AgeIntervalDG
from utils.utils import get_tra_val_tes_size, from_pickle, to_pickle


def ask_for_relaxed_age():
    relaxed_ages = pickle.load(open("{}in\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    while True:
        img = input("Enter the face idx [0, {}]: ".format(len(relaxed_ages)))
        print('{}.png: {}'.format(img, relaxed_ages[int(img)]))


def testDG(data_generator, set_size, batch_size, uni_tra):
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

    print("{} / {}".format(samples_parsed, num_batches * batch_size))
    assert samples_parsed == num_batches * batch_size
    print('Test passed.')


def testMemoryLeak(data_generator, set_size, batch_size):
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


def testDifferentIdxs(testname, idxs1, idxs2):
    for idx1 in idxs1:
        for idx2 in idxs2:
            assert idx1 != idx2

    print("{} OK".format(testname))


def testWIKIAgeDG():
    data_generator_params = {'batch_size': 66,
                             'dim': (160, 160, 3),
                             'embedding_size': 64,
                             'dataset_path': "{}18_58_copy\\".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS),
                             'img_format': ".png"}

    ages = pickle.load(open('{}18_58_copy\\ages.pickle'.format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))[:3000]
    set_size = len(ages)

    print(set_size)

    tra_size, val_size, tes_size = get_tra_val_tes_size(set_size=set_size,
                                                        split_train_val=90,
                                                        split_train_test=90)

    train_generator = AgeDG(ages=ages[0:tra_size],
                            set_size=tra_size,
                            **data_generator_params)
    validation_generator = AgeDG(ages=ages[tra_size:tra_size + val_size],
                                 set_size=val_size,
                                 **data_generator_params)
    test_generator = AgeDG(ages=ages[tra_size + val_size:],
                           set_size=tes_size,
                           **data_generator_params)

    print('Training DataGenerator')
    testDG(data_generator=train_generator,
           set_size=tra_size,
           batch_size=data_generator_params["batch_size"],
           uni_tra=0)

    print('Validation DataGenerator')
    testDG(data_generator=validation_generator,
           set_size=val_size,
           batch_size=data_generator_params["batch_size"],
           uni_tra=0)

    print('Testing DataGenerator')
    testDG(data_generator=test_generator,
           set_size=tes_size,
           batch_size=data_generator_params["batch_size"],
           uni_tra=0)


def testIMDBAgeDG():
    data_generator_params = {'batch_size': 66,
                             'dim': (160, 160, 3),
                             'embedding_size': 64,
                             'dataset_path': IMDB_ALIGNED_MTCNN_160_ABS,
                             'img_format': ".jpg"}

    ages = from_pickle("{}ages.pickle".format(IMDB_ALIGNED_MTCNN_160_ABS))
    tra_gen = AgeDG(ages=ages, set_size=len(ages), **data_generator_params)
    testDG(data_generator=tra_gen,
           set_size=len(ages),
           batch_size=data_generator_params["batch_size"],
           uni_tra=0)


# def testWIKIAgeRelaxedIntervalDG():
#     data_generator_params = {'batch_size': 66,
#                              'dim': (160, 160, 3),
#                              'embedding_size': 128}
#
#     tra_relaxed_ages = pickle.load(open("{}in\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
#     tes_relaxed_ages = pickle.load(open("{}out\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
#     tra_size = len(tra_relaxed_ages)
#     tes_size = len(tes_relaxed_ages)
#
#     train_generator = AgeRelaxedIntervalDG(relaxed_ages=tra_relaxed_ages,
#                                            set_size=tra_size,
#                                            training_flag=1,
#                                            **data_generator_params)
#
#     test_generator = AgeRelaxedIntervalDG(relaxed_ages=tes_relaxed_ages,
#                                           set_size=tes_size,
#                                           training_flag=0,
#                                           **data_generator_params)
#
#     print('Training WIKI_Uni_Relaxed_DataGenerator')
#     testDG(train_generator, tra_size, 66, uni_tra=1)
#
#     print('Testing WIKI_Uni_Relaxed_DataGenerator')
#     testDG(test_generator, tes_size, 66, uni_tra=0)


if __name__ == "__main__":
    # testWIKIAgeDG()
    data_generator_params = {'batch_size': 66,
                             'embedding_size': 128,
                             'dataset_path': "{}first_100k\\".format(IMDB_ALIGNED),
                             'img_format': ".jpg",
                             'img_dimension': (160, 160, 3)}


    eigenvectors = from_pickle("{}0_39999_eigenvectors_normalized.pickle".format(data_generator_params["dataset_path"]))
    print(len(eigenvectors))
    dge = EigenvectorsDG(eigenvectors=eigenvectors,
                         set_size=len(eigenvectors),
                         **data_generator_params)
    testDG(dge, len(eigenvectors), batch_size=data_generator_params["batch_size"], uni_tra=0)
    testDG(dge, len(eigenvectors), batch_size=data_generator_params["batch_size"], uni_tra=0)

    print("EigenvaluesDG OK.")




    # ages = from_pickle("{}ages.pickle".format(data_generator_params["dataset_path"]))
    # dga = AgeDG(ages=ages,
    #             set_size=len(ages),
    #             **data_generator_params)
    # testDG(dga, len(ages), batch_size=data_generator_params["batch_size"], uni_tra=0)
    # testDG(dga, len(ages), batch_size=data_generator_params["batch_size"], uni_tra=0)
    #
    # print("AgeDG OK.")




    # ages = from_pickle("{}out\\ages.pickle".format(data_generator_params["dataset_path"]))
    # ages_relaxed = from_pickle("{}out\\ages_relaxed.pickle".format(data_generator_params["dataset_path"]))
    # d = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    # for i in range(len(ages_relaxed)):
    #     d[ages_relaxed[i]].append(i)
    #
    # for i in range(len(ages_relaxed)):
    #     assert i in d[ages_relaxed[i]]
    #
    # to_pickle(obj=d, filepath="{}out\\age_intervals.pickle".format(data_generator_params["dataset_path"]))
    #
    # age_intervals_in = from_pickle("{}in\\age_intervals.pickle".format(data_generator_params["dataset_path"]))
    # in_sz = len(age_intervals_in) * len(age_intervals_in[0])
    # dgai = AgeIntervalDG(age_intervals=age_intervals_in,
    #                      num_i=6,
    #                      uni=1,
    #                      set_size=in_sz,
    #                      **data_generator_params)
    # testDG(dgai, in_sz, batch_size=data_generator_params["batch_size"], uni_tra=0)
    # testDG(dgai, in_sz, batch_size=data_generator_params["batch_size"], uni_tra=0)
    #
    # print("(in) AgeIntervalDG OK.")
    #
    # age_intervals_out = from_pickle("{}out\\ages_relaxed.pickle".format(data_generator_params["dataset_path"]))
    # out_sz = len(age_intervals_out)
    # # for i in range(6):
    # #     out_sz += len(age_intervals_out[i])
    # dgao = AgeIntervalDG(age_intervals=age_intervals_out,
    #                      num_i=6,
    #                      uni=0,
    #                      set_size=out_sz,
    #                      **data_generator_params)
    # testDG(dgao, out_sz, batch_size=data_generator_params["batch_size"], uni_tra=0)
    # testDG(dgao, out_sz, batch_size=data_generator_params["batch_size"], uni_tra=0)
    #
    # print("(out) AgeIntervalDG OK.")
