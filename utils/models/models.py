import tensorflow as tf
from keras import Sequential
import keras
import keras_vggface
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, concatenate, Conv2D, MaxPooling2D, Lambda
from keras.utils import plot_model
from keras.models import load_model

from utils.constants import FACENET_MODEL_ABS, FACENET_WEIGHTS_ABS
from utils.models.facenet import InceptionResNetV1


def create_simple_embeddings_cnn(input_shape, embedding_size):
    model = Sequential(name='embeddings_model')
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(embedding_size, name='embeddings'))

    return model


def create_vggface_embeddings_cnn(input_shape, embedding_size):
    vggface_model = keras_vggface.VGGFace(model='vgg16')
    return vggface_model


def create_vgg16_embeddings_cnn(input_shape, embedding_size):
    vgg16_model = keras.applications.vgg16.VGG16()
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    # for layer in model.layers:
    #     layer.trainable = False

    model.add(Dense(embedding_size, name='embeddings'))

    return model


def create_facenet_nn3_embeddings_cnn(input_shape, embedding_size):
    # facenet = load_model(FACENET_MODEL_ABS)
    # facenet.load_weights(FACENET_WEIGHTS_ABS)
    # print(type(facenet))
    # m = Sequential(facenet.layers)
    # print(type(m))
    return InceptionResNetV1(weights_path=FACENET_WEIGHTS_ABS)

def create_concatenated_model(args):
    cnn_type_to_function = {
        "simple": create_simple_embeddings_cnn,
        "vgg16": create_vgg16_embeddings_cnn,
        "vggface": create_vggface_embeddings_cnn,
        "facenet": create_facenet_nn3_embeddings_cnn
    }

    label_size = 1 if args.criterion == "age" else args.n_eigenvalues

    input_shape = (args.image_size, args.image_size, args.n_image_channels)

    try:
        embeddings_network = cnn_type_to_function[args.embeddings_cnn](input_shape=input_shape,
                                                                       embedding_size=args.embedding_size)

        embeddings_network.summary()

        # plot_model(embeddings_network, to_file='experiments/{}_embeddings_model.png'.format(embeddings_cnn),
        #            show_shapes=True, show_layer_names=True, dpi=192)

        input_images = Input(shape=input_shape, name='input_image')  # input layer for images
        input_labels = Input(shape=(label_size,), name='input_label')  # input layer for labels
        embeddings = embeddings_network([input_images])  # output of network -> embeddings
        labels_plus_embeddings = concatenate([input_labels, embeddings],
                                             name='label_embedding')  # concatenating the labels + embeddings

        # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
        concatenated_model = Model(inputs=[input_images, input_labels], outputs=labels_plus_embeddings,
                                   name='concatenated_network')

        # plot_model(concatenated_model, to_file='experiments/final_model.png',
        #            show_shapes=True, show_layer_names=True, dpi=192)

        return concatenated_model
    except KeyError:
        raise KeyError("The embeddings_cnn '{}' is not supported."
                       " Supported embeddings_cnns are: 'simple', 'vgg16'.".format(cnn_type_to_function))


if __name__ == "__main__":
    m = create_facenet_nn3_embeddings_cnn(None, None)
    # m.summary()

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    #     tf.keras.layers.MaxPooling2D(pool_size=2),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=2),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation=None),  # No activation on final dense layer
    #     tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    # ])
    #
    # model.summary()