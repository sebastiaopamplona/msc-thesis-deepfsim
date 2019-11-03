from sklearn import preprocessing
import io
import os
import argparse
import pickle

from keras.optimizers import Adam

from utils.constants import WIKI_ALIGNED_MTCNN_160_ABS, WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS
from utils.data.data_generators import WIKI_DataGenerator, WIKI_Uni_Relaxed_DataGenerator
from utils.loss_functions.semihard_triplet_loss import adapted_semihard_triplet_loss
from utils.models.models import create_concatenated_model
from utils.utils import get_tra_val_tes_size, get_tra_val_tes_idxs, calc_mean_and_std


def print_fit_details(args, set_size, train_size, val_size, test_size):
    """
    Prints the details of a training session.
    """
    print('Criterion:\t\t\t{}'.format(args.criterion))
    print('Triplet strategy:\t{}'.format(args.triplet_strategy))
    print('CNN:\t\t\t\t{}'.format(args.embeddings_cnn))
    print('Embedding size:\t\t{}'.format(args.embedding_size))
    print('Number of epochs:\t{}'.format(args.num_epochs))
    print('Batch size:\t\t\t{}'.format(args.batch_size))
    print('Shuffle:\t\t\t{}'.format(bool(args.shuffle)))
    print('Faces aligned:\t\t{}'.format(bool(args.face_aligned)))
    if args.criterion == "age":
        print('Age scoped:\t\t\t{}'.format(bool(args.age_scoped)))
        print('Age relaxed:\t\t{}'.format(bool(args.age_relaxed)))
        print('Age interval:\t\t{}'.format(bool(args.age_interval)))
    print('Uniformized:\t\t{}'.format(bool(args.uniformized)))
    print('Training size:\t\t{}'.format(train_size))
    print('Dataset size:\t\t{}'.format(set_size))
    print('Validation size:\t{}'.format(val_size))
    print('Testing size:\t\t{}'.format(test_size))


def get_args():
    """
    Argument parser.

    es_<EMBEDDING SIZE>_
    e_<NUMBER OF EPOCHS>_
    bs_<BATCH SIZE>_
    ts_<TRAINSIZE>_
    s_<SHUFFLE (0/1)>_
    as_<AGE SCOPED (0/1)>_
    ar_<AGE RELAXED (0/1)>_
    ai_<AGE INTERVAL (0: none/1: relaxed/2: 5in5/3: 10in10)>_
    fa_<FACES ALIGNED (0/1)>
    """
    parser = argparse.ArgumentParser()

    # TRAIN?
    parser.add_argument(
        '--train',
        type=int,
        required=True,
        help='0/1 flag indicating if the model is to train; if 0, the model will predict and create'
             'the necessary files for tensorboard')

    # DATASET PATH
    parser.add_argument(
        '--dataset-path',
        type=str,
        # required=True,
        default=WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS,
        help='path to the dataset, default=utils.constants.WIKI_ALIGNED_MTCNN_PATH_ABS')

    # EMBEDDING SIZE
    parser.add_argument(
        '--embedding-size',
        type=int,
        default=128,
        help='size of the embeddings, default=64')

    # NUMBER OF EPOCHS
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=5,
        help='number of times to train on the data, default=5')

    # BATCH SIZE
    parser.add_argument(
        '--batch-size',
        type=int,
        default=66, # to try with 6 classes of relaxed interval age (11 of each in each batch)
        # default=70,
        help='number of records to read during each training step, default=32')

    # SHUFFLE
    parser.add_argument(
        '--shuffle',
        type=int,
        default=1,
        help='flag (0/1) indicating if the dataset is being shuffled during training, default=1')

    # AGE SCOPED
    parser.add_argument(
        '--age-scoped',
        type=int,
        default=1,
        help='flag (0/1) indicating if the training is being done on a scoped age dataset, default=1')

    # AGE RELAXED
    parser.add_argument(
        '--age-relaxed',
        type=int,
        default=0,
        help='flag (0/1) indicating if the training is being done with relaxed age triplet selection, default=1')

    # AGE INTERVAL
    parser.add_argument(
        '--age-interval',
        type=int,
        default=1,
        help='flag (0: none | 1: relaxed | 2: 5in5 | 3: 10in10) indicating if the training is being done on a'
             'age dataset divided into age intervals, default=0')

    # UNIFORMIZED
    parser.add_argument(
        '--uniformized',
        type=int,
        default=1,
        help='flag (0/1) indicating if the training is being done on uniformed distributed data, default=1')

    # FACES ALIGNED
    parser.add_argument(
        '--face-aligned',
        type=int,
        default=1,
        help='flag (0/1) indicating if the the dataset has the face images aligned by the eyes, default=1')

    # EMBEDDINGS CNN
    parser.add_argument(
        '--embeddings-cnn',
        default='facenet',
        type=str,
        help='embeddings cnn id (eg. vgg16, facenet, small_cnn)')

    # LEARNING RATE
    parser.add_argument(
        '--learning-rate',
        default=1e-4,
        type=float,
        help='learning rate for gradient descent, default=1e-4')

    # IMAGE SIZE
    parser.add_argument(
        '--image-size',
        type=int,
        default=160,
        help='height of the input images (images are squared, height=width), default=224')

    # NUMBER OF IMAGE CHANNELS
    parser.add_argument(
        '--n-image-channels',
        type=int,
        default=3,
        help='number of channels of the image (rgb = 3), default=3')

    # SIMILARITY CRITERION
    parser.add_argument(
        '--criterion',
        type=str,
        default='age',
        help='similarity criterion')

    # TRIPLET STRATEGY
    parser.add_argument(
        '--triplet-strategy',
        type=str,
        default='adapted_semihard',
        help='triplet mining strategy, default=adapted_semihard')

    args, _ = parser.parse_known_args()

    return args


def get_ages(age_interval):
    # no interval
    if age_interval == 0:
        return pickle.load(open('{}ages.pickle'.format(args.dataset_path), 'rb'))
    # relaxed interval
    elif age_interval == 1:
        return pickle.load(open('{}ages_relaxed.pickle'.format(args.dataset_path), 'rb'))
    # TODO: 5in5 and 10in10
    # default
    else:
        return pickle.load(open('{}ages.pickle'.format(args.dataset_path), 'rb'))


def get_model_name(args, tra_size):
    return "es_{}_e_{}_bs_{}_ts_{}_s_{}_as_{}_ar_{}_ai_{}_u_{}_fa_{}.h5".format(args.embedding_size,
                                                                                args.num_epochs,
                                                                                args.batch_size,
                                                                                tra_size,
                                                                                args.shuffle,
                                                                                args.age_scoped,
                                                                                args.age_relaxed,
                                                                                args.age_interval,
                                                                                args.uniformized,
                                                                                args.face_aligned)


if __name__ == '__main__':
    args = get_args()

    # Create the model
    model = create_concatenated_model(args)
    model.compile(loss=adapted_semihard_triplet_loss,
                  metrics=['accuracy'],
                  optimizer=Adam(lr=args.learning_rate))
    model.summary()

    # Configuring the DataGenerator for the training and validation set
    '''
    # Before testing WIKI_Uni_Relaxed_DataGenerator
    
    data_gen_params = {'batch_size': args.batch_size,
                       'dim': (args.image_size, args.image_size, args.n_image_channels),
                       'embedding_size': args.embedding_size}

    ages = get_ages(args.age_interval)
    print(ages)
    set_size = len(ages)

    tra_size, val_size, tes_size = get_tra_val_tes_size(set_size=set_size,
                                                        split_train_val=90,
                                                        split_train_test=90)

    train_generator = WIKI_DataGenerator(ages=ages[0:tra_size], set_size=tra_size, **data_gen_params)
    validation_generator = WIKI_DataGenerator(ages=ages[tra_size:tra_size + val_size], set_size=val_size, **data_gen_params)
    test_generator = WIKI_DataGenerator(ages=ages[tra_size + val_size:], set_size=tes_size, **data_gen_params)
    '''
    data_generator_params = {'batch_size': 66,
                             'dim': (160, 160, 3),
                             'embedding_size': 128}

    tra_relaxed_ages = pickle.load(
        open("{}in\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
    tes_relaxed_ages = pickle.load(
        open("{}out\\relaxed_ages.pickle".format(WIKI_ALIGNED_MTCNN_UNI_RELAXED_160_ABS), 'rb'))
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

    experiments_path = "experiments/{}/{}/".format(args.criterion, args.triplet_strategy)
    model_path = "{}models/{}/".format(experiments_path, args.embeddings_cnn)
    model_name = get_model_name(args, tra_size)

    print(model_path + model_name)
    print_fit_details(args, tra_size + tes_size, tra_size, 0, tes_size)
    '''
    print_fit_details(args, set_size, tra_size, val_size, tes_size)
    '''
    # [Below is tested.]
    if args.train:
        # Train the model
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=int(tra_size / args.batch_size) - 1,
                            # validation_data=validation_generator,
                            epochs=args.num_epochs,
                            max_queue_size=1,
                            verbose=1)

        # Save the weights
        model.save_weights(model_path + model_name)
        
    else:
        # Load the weights from the trained model and produce the embeddings for the test set
        print("Predicting with the weights from: {}".format(model_path + model_name))
        model.load_weights(model_path + model_name)
        embeddings = model.predict_generator(generator=test_generator,
                                             steps=int(tes_size / args.batch_size) - 1,
                                             # callbacks=None,
                                             max_queue_size=1,
                                             # workers=1,
                                             verbose=1)[:, 1:]

        # (not normalized)
        # mean: 0.4748522024777638 | std: 0.3036557089186148
        # m, s = calc_mean_and_std(embeddings)
        # print("(not normalized) mean: {} | std: {}".format(m, s))

        embeddings_path = "{}embeddings/{}/{}/".format(experiments_path, args.embeddings_cnn, model_name.split(".")[0])

        normalized_embeddings = preprocessing.normalize(embeddings, norm="l2")

        # m, s = calc_mean_and_std(normalized_embeddings)
        # print("(normalized) mean: {} | std: {}".format(m, s))

        try:
            os.mkdir(embeddings_path)
        except FileExistsError:
            pass

        # Create the two necessary .tsv files for the Tensorboard visualization
        out_embeddings = io.open("{}embeddings.tsv".format(embeddings_path),
                                 'w',
                                 encoding='utf-8')
        out_normalized_embeddings = io.open("{}normalized_embeddings.tsv".format(embeddings_path),
                                            'w',
                                            encoding='utf-8')
        out_ages = io.open("{}ages.tsv".format(embeddings_path),
                           'w',
                           encoding='utf-8')
        out_filenames = io.open("{}ages_filenames.tsv".format(embeddings_path),
                                'w',
                                encoding='utf-8')
        # out_relaxed_ages = io.open("{}relaxed_ages.tsv".format(embeddings_path),
        #                    'w',
        #                    encoding='utf-8')
        # out_relaxed_filenames = io.open("{}relaxed_ages_filenames.tsv".format(embeddings_path),
        #                         'w',
        #                         encoding='utf-8')


        # ages = get_ages(0)
        # relaxed_ages = get_ages(1)
        # print(ages)
        # print(relaxed_ages)

        for i in range(embeddings.shape[0]):
            out_normalized_embeddings.write('{}\n'.format('\t'.join([str(x) for x in normalized_embeddings[i][:]])))
            out_embeddings.write('{}\n'.format('\t'.join([str(x) for x in embeddings[i][:]])))
            out_ages.write('{}\n'.format(tes_relaxed_ages[i]))
            # out_ages.write('{}\n'.format(ages[tra_size + val_size + i]))
            out_filenames.write('{}|{}.png\n'.format(tes_relaxed_ages[i], i))
            # out_filenames.write('{}|{}.png\n'.format(ages[tra_size + val_size + i], tra_size + val_size + i))
            # out_relaxed_ages.write('{}\n'.format(relaxed_ages[tra_size + val_size + i]))
            # out_relaxed_filenames.write('{}|{}.png\n'.format(relaxed_ages[tra_size + val_size + i], tra_size + val_size + i))

