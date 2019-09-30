from keras import Sequential

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, concatenate, Conv2D, MaxPooling2D
from keras.utils import plot_model


def create_embeddings_model(input_shape, embedding_size):
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
    model.add(Dense(embedding_size))

    return model


def create_concatenated_model(params):
    input_shape = (params.image_size, params.image_size, params.n_image_channels)

    embeddings_network = create_embeddings_model(input_shape=input_shape,
                                                 embedding_size=params.embedding_size)

    plot_model(embeddings_network, to_file='experiments/embeddings_model.png',
               show_shapes=True, show_layer_names=True, dpi=192)

    input_images = Input(shape=input_shape, name='input_image')  # input layer for images
    input_labels = Input(shape=(1,), name='input_label')  # input layer for labels
    embeddings = embeddings_network([input_images])  # output of network -> embeddings
    labels_plus_embeddings = concatenate([input_labels, embeddings],
                                         name='label_embedding')  # concatenating the labels + embeddings

    # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
    concatenated_model = Model(inputs=[input_images, input_labels], outputs=labels_plus_embeddings,
                               name='concatenated_network')

    plot_model(concatenated_model, to_file='experiments/final_model.png',
               show_shapes=True, show_layer_names=True, dpi=192)

    return concatenated_model
