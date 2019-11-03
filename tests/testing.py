import cv2
import pickle
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from utils.constants import WIKI_ALIGNED_160_ABS, WIKI_ALIGNED_MTCNN_160_ABS, DESKTOP_PATH_ABS

ages = pickle.load(open('{}ages.pickle'.format(WIKI_ALIGNED_160_ABS), 'rb'))

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


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


test_face = WIKI_ALIGNED_MTCNN_160_ABS + "0.png"
print(test_face)

mtcnned = extract_face(test_face)
cv2.imwrite(DESKTOP_PATH_ABS + "0.png", mtcnned)

