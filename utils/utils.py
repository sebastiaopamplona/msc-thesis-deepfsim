import argparse
import numpy
import statistics
import math
import pickle
import random
import keras

from PIL import Image

from utils.constants import WIKI_18_58_160, WIKI_18_58_224, WIKI_ALIGNED_UNI_160, \
    WIKI_AUGMENTED_UNI_160, IMDB_ALIGNED
from utils.data.data_generators import AgeDG, AgeIntervalDG, EigenvectorsDG


def get_tra_val_tes_size(set_size, split_train_val, split_train_test):
    """
    Calculates the sizes of the training, validation and test sets.
    :param set_size: total set size
    :param split_train_val: split for train/validation
    :param split_train_test: split for train/test

    :return: the size of training/validation/test size, accorind
    to the split proportions
    """

    train_size = int(split_train_test * .01 * set_size)
    test_size = int(set_size - train_size)

    train_size = int(split_train_val * .01 * train_size)
    val_size = set_size - train_size - test_size

    return train_size, val_size, test_size


def to_pickle(obj, filepath):
    """
    Writes an object to a pickle file.

    :param obj: object to be written
    :param filepath: filepath of the .pickle file
    """

    pickle_out = open(filepath, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def from_pickle(filepath):
    """
    Reads an object from a pickle file.

    :param filepath: filepath of the .pickle file

    :return: object read
    """

    return pickle.load(open(filepath, 'rb'))


def get_mean_and_std(embeddings, use_numpy=False):
    """
    Calculates the mean and the std of the euclidean distances
    between all embeddings. To be used in the Recursive Lists of
    Clusters metric data structure.

    :param embeddings: embeddings
    :param use_numpy: flag indicating the usage of numpy

    :return: mean and std
    """

    size = len(embeddings)
    distances = []
    for i in range(size):
        for j in range(i, size):
            print("{}-{}".format(i, j))
            if use_numpy:
                distances.append(numpy.linalg.norm(embeddings[i] - embeddings[j]))
            else:
                distances.append(math.sqrt(sum([(a - b) ** 2
                                                for a, b in zip(embeddings[i],
                                                                embeddings[j])])))

    return statistics.mean(distances), statistics.stdev(distances)


def get_label_distribution(labels):
    """
    Calculates the label ditribution of a dataset.

    :param labels: dataset labels

    :return: dictionary with the label distribution
    """

    distr = {}
    for label in labels:
        try:
            ctr = distr[label]
            distr[label] = ctr + 1
        except KeyError as e:
            distr[label] = 1

    return distr


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
        help='0/1 flag indicating if the model is to train; if 0, the model '
             'will predict and create the necessary files for tensorboard')

    # DATASET PATH
    parser.add_argument(
        '--dataset-path',
        type=str,
        # required=True,
        default="{}first_100k\\".format(IMDB_ALIGNED),
        help='path to the dataset, '
             'default=utils.constants.WIKI_ALIGNED_MTCNN_PATH_ABS')

    # DATASET
    parser.add_argument(
        '--dataset',
        type=str,
        default="imdb",
        help='name of the dataset, '
             'default=wiki')

    # EMBEDDING SIZE
    parser.add_argument(
        '--embedding-size',
        type=int,
        default=128,
        help='size of the embeddings, '
             'default=64')

    # NUMBER OF EPOCHS
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1,
        help='number of times to train on the data, '
             'default=20')

    # BATCH SIZE
    parser.add_argument(
        '--batch-size',
        type=int,
        default=66,
        help='number of records to read during each training step, '
             'default=66')

    # SHUFFLE
    parser.add_argument(
        '--shuffle',
        type=int,
        default=1,
        help='flag (0/1) indicating if the dataset is being shuffled during '
             'training, '
             'default=1')

    # AGE SCOPED
    parser.add_argument(
        '--age-scoped',
        type=int,
        default=1,
        help='flag (0/1) indicating if the training is being done on a scoped '
             'age dataset, '
             'default=1')

    # AGE RELAXED
    parser.add_argument(
        '--age-relaxed',
        type=int,
        default=1,
        help='flag (0/1) indicating if the training is being done with relaxed '
             'age triplet selection, '
             'default=1')

    # AGE INTERVAL
    parser.add_argument(
        '--age-interval',
        type=int,
        default=0,
        help='flag (0: none | 1: relaxed | 2: 5in5 | 3: 10in10) indicating if '
             'the training is being done on a age dataset divided into age '
             'intervals, '
             'default=0')

    # UNIFORMIZED
    parser.add_argument(
        '--uniformized',
        type=int,
        default=0,
        help='flag (0/1) indicating if the training is being done on uniformed '
             'distributed data, '
             'default=1')

    # FACES ALIGNED
    parser.add_argument(
        '--face-aligned',
        type=int,
        default=1,
        help='flag (0/1) indicating if the the dataset has the face images '
             'aligned by the eyes, '
             'default=1')

    # EMBEDDINGS CNN
    parser.add_argument(
        '--embeddings-cnn',
        default='facenet',
        type=str,
        help='embeddings cnn id (eg. vgg16, facenet, small_cnn), '
             'default=facenet')

    # OPTIMIZER
    parser.add_argument(
        '--optimizer',
        default="Adam",
        type=str,
        help='optimizer (eg. Adam, SGD, RMSprop (CASE SENSITIVE!)), '
             'default=Adam')

    # LEARNING RATE
    parser.add_argument(
        '--learning-rate',
        default=1e-4,
        type=float,
        help='learning rate for gradient descent, '
             'default=1e-4')

    # IMAGE SIZE
    parser.add_argument(
        '--image-size',
        type=int,
        default=160,
        help='height of the input images (images are squared, height=width), '
             'default=224')

    # IMAGE FORMAT
    parser.add_argument(
        '--image-format',
        type=str,
        default=".jpg",
        help='image format, '
             'defualt=.png')

    # NUMBER OF IMAGE CHANNELS
    parser.add_argument(
        '--n-image-channels',
        type=int,
        default=3,
        help='number of channels of the image (rgb = 3), '
             'default=3')

    # SIMILARITY CRITERION
    parser.add_argument(
        '--criterion',
        type=str,
        default='eigenvectors',
        help='similarity criterion (supported: age, eigenvectors), '
             'default=age')

    # NUMBER OF EIGENVALUES
    parser.add_argument(
        '--n-eigenvectors',
        type=str,
        default=250,
        help='length of the eigenvectors, '
             'default=250')

    # TRIPLET STRATEGY
    parser.add_argument(
        '--triplet-strategy',
        type=str,
        default='adapted_semihard',
        help='triplet mining strategy, '
             'default=adapted_semihard')

    args, _ = parser.parse_known_args()

    return args


def get_parameters_details(args, set_size, tra_sz, val_sz, tes_sz):
    """Returns a string with the details of a training session."""
    details = ""
    details += '[Criterion]: {}\n'.format(args.criterion)
    if args.criterion == "age":
        details += '[Age scoped]: {}\n'.format(bool(args.age_scoped))
        details += '[Age relaxed]: {}\n'.format(bool(args.age_relaxed))
        details += '[Age interval]: {}\n'.format(bool(args.age_interval))
    elif args.criterion == "eigenvectors":
        thresholds = {"0": 0.3, "1": 0.5, "2": 0.8, "3": 1.0, "4": 1.3}
        fr = open("eigenvectors_threshold.txt", "r")
        counter = fr.read()
        fr.close()
        threshold = thresholds[counter]
        details += "[Eigenvectos similarity threshold]: {}\n".format(threshold)

    details += "\n"
    details += '[Dataset size]: {}\n'.format(set_size)
    details += '[Training size]: {}\n'.format(tra_sz)
    details += '[Validation size]: {}\n'.format(val_sz)
    details += '[Testing size]: {}\n'.format(tes_sz)
    details += '[Faces aligned]: {}\n'.format(bool(args.face_aligned))
    details += '[Uniformized]: {}\n'.format(bool(args.uniformized))
    details += "\n"
    details += '[Embedding size]: {}\n'.format(args.embedding_size)
    details += '[CNN]: {}\n'.format(args.embeddings_cnn)
    details += '[Triplet strategy]: {}\n'.format(args.triplet_strategy)
    details += '[Number of epochs]: {}\n'.format(args.num_epochs)
    details += '[Batch size]: {}\n'.format(args.batch_size)
    details += '[Optimizer]: {}\n'.format(args.optimizer)
    details += '[Learning rate]: {}\n'.format(args.learning_rate)
    details += '[Shuffle]: {}\n'.format(bool(args.shuffle))
    details += "\n"
    experiments_path = "experiments/{}/{}/{}/".format(args.dataset,
                                                      args.criterion,
                                                      args.triplet_strategy)
    model_path = "{}models/{}/".format(experiments_path, args.embeddings_cnn)
    model_name = get_model_name(args, tra_sz)
    details += '[Model name]: {}\n'.format(model_name)
    details += '[Experiments path]: {}\n'.format(experiments_path)
    details += '[Model path]: {}\n'.format(model_path)

    return experiments_path, model_path, model_name, details


def print_parameters(args, set_size, tra_sz, val_sz, tes_sz):
    """Prints the details of a training session."""

    print('Criterion:\t\t{}'.format(args.criterion))
    print('Triplet strategy:\t{}'.format(args.triplet_strategy))
    print('CNN:\t\t\t{}'.format(args.embeddings_cnn))
    print('Embedding size:\t\t{}'.format(args.embedding_size))
    print('Number of epochs:\t{}'.format(args.num_epochs))
    print('Optimizer:\t\t{}'.format(args.optimizer))
    print('Learning rate:\t\t{}'.format(args.learning_rate))
    print('Batch size:\t\t\t{}'.format(args.batch_size))
    print('Shuffle:\t\t{}'.format(bool(args.shuffle)))
    print('Faces aligned:\t\t{}'.format(bool(args.face_aligned)))
    if args.criterion == "age":
        print('Age scoped:\t\t{}'.format(bool(args.age_scoped)))
        print('Age relaxed:\t\t{}'.format(bool(args.age_relaxed)))
        print('Age interval:\t\t{}'.format(bool(args.age_interval)))
    print('Uniformized:\t\t{}'.format(bool(args.uniformized)))
    print('Training size:\t\t{}'.format(tra_sz))
    print('Dataset size:\t\t{}'.format(set_size))
    print('Validation size:\t{}'.format(val_sz))
    print('Testing size:\t\t{}'.format(tes_sz))

    experiments_path = "experiments/{}/{}/{}/".format(args.dataset,
                                                      args.criterion,
                                                      args.triplet_strategy)
    model_path = "{}models/{}/".format(experiments_path, args.embeddings_cnn)
    model_name = get_model_name(args, tra_sz)
    print()
    print('Model name:\t\t\t{}'.format(model_name))
    print('Experiments path:\t{}'.format(experiments_path))
    print('Model path:\t\t\t{}'.format(model_path))

    return experiments_path, model_path, model_name


def get_in_out_labels(args):
    """
    Returns the labels of the dataset inside and outside the label uniform
    distribution.

    :param args: args

    :return: in_labels, in_sz, out_labels, out_sz
    """
    if args.age_interval == 1:
        in_labels = from_pickle("{}in\\age_intervals.pickle"
                                .format(args.dataset_path))

        in_keys = list(in_labels.keys())
        # this is possible because all dict values (label lists)
        # contain the sames size, due to the uniform distribution
        in_sz = len(in_keys) * len(in_labels[in_keys[0]])

        out_labels = from_pickle("{}out\\ages_relaxed.pickle"
                                 .format(args.dataset_path))

        # out_labels is a list, not a dict
        return in_labels, in_sz, out_labels, len(out_labels)
    else:
        raise Exception('Interval {} not supported'.format(args.age_interval))


def get_optimizers_dict(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False,
                        momentum=0.9, nesterov=False,
                        rho=0.9):
    """

    For Adam
    :param learning_rate: learning rate
    :param beta_1: float, 0 < beta < 1; generally close to
    :param beta_2: float, 0 < beta < 1; generally close to 1
    :param amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm from the paper
                    "On the Convergence of Adam and Beyond"

    For SGD
    :param momentum: float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations
    :param nesterov: boolean; whether to apply Nesterov momentum

    For RMSprop
    :param rho: float >= 0

    :return: a dictionary mapping the name of the optimizer to the keras.optimizer
    """

    assert 0.0 < beta_1 < 1.
    assert 0.0 < beta_2 < 1.
    assert 0.0 <= momentum
    assert 0.0 <= rho

    return {
        "Adam": keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad),
        "SGD": keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=True),
        "RMSprop": keras.optimizers.RMSprop(lr=learning_rate, rho=rho)
    }


def get_labels(args):
    if args.criterion == "age":
        if args.age_interval == 0:
            labels = from_pickle('{}ages.pickle'.format(args.dataset_path))
        else:
            raise Exception('Wrong function for interval {}; '
                            'call function get_in_out_labels'
                            .format(args.age_interval))
    elif args.criterion == "eigenvectors":
        # labels = from_pickle("{}eigenvectors.pickle".format(args.dataset_path))
        labels = from_pickle("{}0_39999_eigenvectors_normalized.pickle".format(args.dataset_path))
        # eigen_d_keys = list(labels.keys())
        # eigen_l = []
        # for k in eigen_d_keys:
        #     eigen_l.append(labels[k])
        # labels = eigen_l
    else:
        raise Exception('Criterion {} not supported'.format(args.criterion))

    return labels


def get_model_name(args, tra_size):
    return "es_{}_e_{}_bs_{}_ts_{}_s_{}_as_{}_ar_{}_ai_{}_u_{}_fa_{}.h5"\
            .format(args.embedding_size,
                    args.num_epochs,
                    args.batch_size,
                    tra_size,
                    args.shuffle,
                    args.age_scoped,
                    args.age_relaxed,
                    args.age_interval,
                    args.uniformized,
                    args.face_aligned)


def get_argsDG(args):
    return {"batch_size": args.batch_size,
            "embedding_size": args.embedding_size,
            "dataset_path": args.dataset_path,
            "img_format": args.image_format,
            "img_dimension": (args.image_size,
                              args.image_size,
                              args.n_image_channels)}


def get_path_checkpoints(model_path, model_name):
    return "{}{}/checkpoints/".format(model_path, model_name.split(".")[0])


def get_path_embeddings(experiments_path, embeddings_cnn, model_name):
    "{}embeddings/{}/{}/".format(experiments_path,
                                 embeddings_cnn,
                                 model_name.split(".")[0])


def get_path_experiments(args):
    return "experiments/{}/{}/{}/".format(args.dataset,
                                          args.criterion,
                                          args.triplet_strategy)


def get_path_model(experiments_path, args):
    return "{}models/{}/".format(experiments_path, args.embeddings_cnn)


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


def get_standardized_pixels(filename):
    """
    Standardizes image pixels; Facenet expects standardized pixels as input.

    :param filename: filepath to the image

    :return: standardized pixels
    """
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = numpy.asarray(image)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    return (pixels - mean) / std


def get_dgs(labels, args, argsDG, tra_sz=None, val_sz=None, tes_sz=None,
            labels_in=None, labels_out=None):
    if args.criterion == "age":
        if args.age_interval == 0:
            tra_dg = AgeDG(ages=labels[0:tra_sz], set_size=tra_sz, **argsDG)
            val_dg = AgeDG(ages=labels[tra_sz:tra_sz + val_sz],
                           set_size=val_sz,
                           **argsDG)
            tes_dg = AgeDG(ages=labels[tra_sz + val_sz:],
                           set_size=tes_sz,
                           **argsDG)
        elif args.age_interval == 1:
            tra_labels_in = {i: labels_in[i][:int(tra_sz / len(labels_in))]
                             for i in range(len(labels_in))}
            val_labels_in = {i: labels_in[i][int(tra_sz / len(labels_in)):]
                             for i in range(len(labels_in))}
            tra_dg = AgeIntervalDG(age_intervals=tra_labels_in,
                                   num_i=len(labels_in),
                                   uni=args.uniformized,
                                   set_size=tra_sz,
                                   **argsDG)

            val_dg = AgeIntervalDG(age_intervals=val_labels_in,
                                   num_i=len(labels_in),
                                   uni=args.uniformized,
                                   set_size=val_sz,
                                   **argsDG)

            tes_dg = AgeIntervalDG(age_intervals=labels_out,
                                   num_i=len(labels_in),
                                   uni=0,
                                   set_size=len(labels_out),
                                   **argsDG)

        else:
            raise Exception('Interval {} not supported'.format(args.age_interval))

    elif args.criterion == "eigenvectors":
        tra_dg = EigenvectorsDG(eigenvectors=labels[0:tra_sz],
                                set_size=tra_sz,
                                **argsDG)
        val_dg = EigenvectorsDG(eigenvectors=labels[tra_sz:tra_sz + val_sz],
                                set_size=val_sz,
                                **argsDG)
        tes_dg = EigenvectorsDG(eigenvectors=labels[tra_sz + val_sz:],
                                set_size=tes_sz,
                                **argsDG)
    else:
        raise Exception('Criterion {} not supported'.format(args.criterion))

    return tra_dg, val_dg, tes_dg


def get_tra_val_tes_idxs(dataset_size, train_size, val_size):
    # TODO: delete
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


def get_ages(age_interval, dataset_path):
    # TODO: delete
    # no interval
    if age_interval == 0:
        return pickle.load(open('{}ages.pickle'.format(dataset_path), 'rb'))
    # relaxed interval
    elif age_interval == 1:
        return pickle.load(open("{}relaxed_ages.pickle".format(dataset_path), 'rb'))
    # TODO: 5in5 and 10in10
    # default
    else:
        return pickle.load(open('{}ages.pickle'.format(dataset_path), 'rb'))
