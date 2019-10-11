import math
import pickle
import numpy as np
import sys
from train import get_tra_val_tes_size
from utils.data_generators import WIKI_DataGenerator


def test_data_generator(data_generator, set_size, batch_size):
    """Simulates the generation of the batches."""
    num_batches = math.ceil(set_size / batch_size)
    samples_parsed = 0

    for i in range(num_batches):
        bx, _ = data_generator.__getitem__(index=0)
        print('Batch [{}]: {} samples'.format(i, len(bx[0])))
        samples_parsed += len(bx[0])

    assert samples_parsed == set_size
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



data_generator_params = {'batch_size': 200,
                         'dim': (224, 224, 3),
                         'embedding_size': 64}

dataset_path = '/home/sebastiao/Desktop/PERS/DEV/github/master-thesis/wiki_age/dataset/mtcnn_extracted/'
ages = pickle.load(open('{}ages.pickle'.format(dataset_path), 'rb'))
set_size = len(ages)

train_size, val_size, test_size = get_tra_val_tes_size(set_size=set_size,
                                                       split_train_val=90,
                                                       split_train_test=90)

train_generator = WIKI_DataGenerator(ages=ages, start_idx=0, set_size=train_size, **data_generator_params)
# validation_generator = WIKI_DataGenerator(ages=ages, start_idx=train_size, set_size=val_size, **data_generator_params)








# print('Training DataGenerator')
# test_data_generator(data_generator=train_generator,
#                     set_size=train_size,
#                     batch_size=200)
#
# print('Validation DataGenerator')
# test_data_generator(data_generator=validation_generator,
#                     set_size=val_size,
#                     batch_size=200)

# print('Training Memory Leak')
# test_memory_leak(data_generator=train_generator,
#                  set_size=train_size,
#                  batch_size=200)

# print(sys.float_info)

