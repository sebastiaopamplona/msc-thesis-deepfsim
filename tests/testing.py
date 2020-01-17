import numpy as np
import cv2
import pickle
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import tensorflow_datasets as tfds
from utils.constants import WIKI_ALIGNED_MTCNN_160_ABS, DESKTOP_PATH_ABS

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import os

from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

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


ages = pickle.load(open('{}ages.pickle'.format(WIKI_ALIGNED_MTCNN_160_ABS), 'rb'))


def ask_for_age():
    while True:
        img = input("Enter the face idx [0, 38781]: ")
        print('{}.png: {} years old'.format(img, ages[int(img)]))


def relaxed_comparison(older_age, younger_age, threshold):
    min_age = int(older_age / (1 + threshold))
    return min_age <= younger_age, min_age


def test_threshold(threshold):
    for age1 in range(0, 99):
        for age2 in range(0, 99):
            rc = relaxed_comparison(max(age1, age2), min(age1, age2), threshold)
            if not rc[0]:
                print('For age {}, the minimum is {}'.format(max(age1, age2), rc[1]))
                break


def resize_face(face, dim=(56, 56)):
    img = cv2.imread(face)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def get_embeddings(filenames):
    # extract faces
    faces = [resize_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    vgg16_model = keras.applications.vgg16.VGG16(input_tensor=Input(shape=(56, 56, 3)))
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    for i in range(len(model.layers) - 2):
        # print(type(model.layers[i]))
        model.layers[i].trainable = False

    model.add(Dense(64, name='embeddings'))
    model.summary()
    # perform prediction
    yhat = model.predict(samples)
    return yhat


if __name__ == "__main__":

    f1 = "100000.jpg"
    filenames = [f1]
    embeddings = get_embeddings(filenames)
