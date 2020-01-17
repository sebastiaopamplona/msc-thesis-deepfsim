import numpy as np
import tensorflow as tf
from keras import Sequential
import keras
import keras_vggface
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, concatenate, Conv2D, MaxPooling2D, \
    Lambda
from keras.utils import plot_model
from keras.models import load_model

from utils.constants import FACENET_MODEL_ABS, FACENET_WEIGHTS_ABS
from utils.models.facenet import InceptionResNetV1
from PIL import Image
from keras.applications.vgg16 import preprocess_input, VGG16


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


def create_vgg16_160_no_top(input_shape, embedding_size):
    vgg16_model = keras_vggface.VGGFace(model='vgg16',
                                        include_top=False,
                                        input_tensor=Input(shape=(160, 160, 3)))

    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    # TODO: figure out if it is necessary to add embedding size layer
    return model


def create_vgg16_embeddings_cnn(input_shape, embedding_size):
    vgg16_model = keras.applications.vgg16.VGG16()
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    for i in range(len(model.layers) - 2):
        # print(type(model.layers[i]))
        model.layers[i].trainable = False

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

    label_size = 1 if args.criterion == "age" else args.n_eigenvectors

    input_shape = (args.image_size, args.image_size, args.n_image_channels)

    try:
        embeddings_network = cnn_type_to_function[args.embeddings_cnn](
            input_shape=input_shape,
            embedding_size=args.embedding_size)

        embeddings_network.summary()

        # plot_model(embeddings_network, to_file='experiments/{}_embeddings_model.png'.format(embeddings_cnn),
        #            show_shapes=True, show_layer_names=True, dpi=192)

        input_images = Input(shape=input_shape, name='input_image')  # input layer for images
        input_labels = Input(shape=(label_size,),
                             name='input_label')  # input layer for labels
        embeddings = embeddings_network([input_images])  # output of network -> embeddings
        labels_plus_embeddings = concatenate([input_labels, embeddings],
                                             name='label_embedding')  # concatenating the labels + embeddings

        # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
        concatenated_model = Model(inputs=[input_images, input_labels],
                                   outputs=labels_plus_embeddings,
                                   name='concatenated_network')

        # plot_model(concatenated_model, to_file='experiments/final_model.png',
        #            show_shapes=True, show_layer_names=True, dpi=192)

        return concatenated_model
    except KeyError:
        raise KeyError("The embeddings_cnn '{}' is not supported."
                       " Supported embeddings_cnns are: 'simple', 'vgg16'.".format(
            cnn_type_to_function))


# DELETE
def get_standardized_pixels(filename, required_size):
    """
    Reads an image and standardizes its pixels; Facenet expects standardized pixels as input.

    :param filename: filepath to the image
    :param required_size: required image size

    :return: standardized pixels
    """
    image = Image.open(filename)
    if image.size[0] != required_size:
        image = image.resize((required_size, required_size))

    image = image.convert('RGB')
    pixels = np.asarray(image)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    return (pixels - mean) / std


# DELETE


if __name__ == "__main__":
    x = create_vgg16_160_no_top(-1, -1)
    x.summary()

    img_data = get_standardized_pixels(
        "C://Users//Sebastião Pamplona//Desktop//DEV//datasets//treated//age//imdb_aligned_mtcnn_160//0.jpg",
        160)

    batch_x = np.zeros((1,
                        160,
                        160,
                        3))
    batch_x[0] = img_data
    # vgg16_feature = x.predict(batch_x)

    model = VGG16(weights='imagenet', include_top=False)
    model.summary()
    vgg16_feature = model.predict(batch_x)
    print(vgg16_feature.flatten())
    print(vgg16_feature.flatten().shape)
    # Feature extraction block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, name='fc6')(x)
    # x = Activation('relu', name='fc6/relu')(x)
    # x = Dense(4096, name='fc7')(x)
    # x = Activation('relu', name='fc7/relu')(x)
    # x = Dense(classes, name='fc8')(x)
    # x = Activation('softmax', name='fc8/softmax')(x)

    '''
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc6 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc6/relu (Activation)        (None, 4096)              0         
    _________________________________________________________________
    fc7 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    fc7/relu (Activation)        (None, 4096)              0         
    _________________________________________________________________
    fc8 (Dense)                  (None, 2622)              10742334  
    _________________________________________________________________
    fc8/softmax (Activation)     (None, 2622)              0 
    '''

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
