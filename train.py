import argparse
import os
import pickle

from keras.optimizers import Adam

from utils.data_generators import WIKI_DataGenerator
from utils.models import create_concatenated_model
from utils.semihard_triplet_loss import adapted_semihard_triplet_loss


def get_tra_val_tes_size(set_size, split_train_val, split_train_test):
    """Aux method to get the sizes of the training, validation and test
    sets."""
    train_size = int(split_train_test * .01 * set_size)
    test_size = int(set_size - train_size)

    train_size = int(split_train_val * .01 * train_size)
    val_size = set_size - train_size - test_size

    return train_size, val_size, test_size

def get_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='path to the WIKI_Age dataset')
    parser.add_argument(
        '--model-dir',
        default='experiments/adapted_semihard',
        type=str,
        help='directory for writing checkpoints and exporting models')
    parser.add_argument(
        '--learning-rate',
        default=1e-4,
        type=float,
        help='learning rate for gradient descent, default=1e-4')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,  # TODO: changed to speed up testing
        help='number of times to go through the data, default=3')
    parser.add_argument(
        '--embedding-size',
        type=int,
        default=64,
        help='size of the embeddings, default=64')
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='height of the input images (images are squared, height=width), default=224')
    parser.add_argument(
        '--n-image-channels',
        type=int,
        default=3,
        help='number of channels of the image (rgb = 3), default=3')
    parser.add_argument(
        '--triplet-strategy',
        type=str,
        default='adapted_semihard',
        help='triplet mining strategy, default=adapted_semihard')

    args, _ = parser.parse_known_args()

    return args

if __name__ == '__main__':
    args = get_args()

    # Create the model
    model = create_concatenated_model(args)
    model.summary()
    model.compile(loss=adapted_semihard_triplet_loss,
                  metrics=['accuracy'],
                  optimizer=Adam(lr=args.learning_rate))

    # Configuring the DataGenerator for the training and validation set
    data_gen_params = {'batch_size': args.batch_size,
                       'dim': (args.image_size, args.image_size, args.n_image_channels),
                       'embedding_size': args.embedding_size}

    ages = pickle.load(open('{}ages.pickle'.format(args.dataset_path), 'rb'))
    set_size = len(ages)

    train_size, val_size, test_size = get_tra_val_tes_size(set_size=set_size,
                                                           split_train_val=90,
                                                           split_train_test=90)

    train_generator = WIKI_DataGenerator(ages=ages, start_idx=0, set_size=train_size, **data_gen_params)
    validation_generator = WIKI_DataGenerator(ages=ages, start_idx=train_size, set_size=val_size, **data_gen_params)

    # Train the model
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=int(train_size / args.num_epochs),
                        validation_data=validation_generator,
                        epochs=args.num_epochs,
                        verbose=1)