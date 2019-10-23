import pickle

dataset_path = 'C:\\Users\\Sebastião Pamplona\\Desktop\\DEV\\datasets\\mtcnn_extracted\\'
ages = pickle.load(open('{}ages.pickle'.format(dataset_path), 'rb'))

def ask_for_age():
    while True:
        img = input("Enter the face idx [0, 38781]: ")
        print('{}.png: {} years old'.format(img, ages[int(img)]))


def relaxed_comparison(older_age, younger_age, threshold):
    min_age = int(older_age / (1 + threshold))
    return min_age <= younger_age, min_age


def test_threshold(threshold):
    for age1 in range(0,99):
        for age2 in range(0,99):
            rc = relaxed_comparison(max(age1, age2), min(age1, age2), threshold)
            if not rc[0]:
                print('For age {}, the minimum is {}'.format(max(age1, age2), rc[1]))
                break


test_threshold(0.15)

import tensorflow as tf


x = [[2]]
m = tf.matmul(x, x)
threshold = 0.15
ages = tf.constant([[17.], [13.], [18.], [77.], [66.], [65.], [99.], [99.]])
min_age = tf.math.round(tf.math.divide(ages, 1 + threshold))

# print("{}".format(ages))
print("{}".format(min_age))
# print("{}".format(tf.transpose(ages)))
# print("{}".format(tf.math.minimum(ages, tf.transpose(ages))))
min_transposed1 = tf.transpose(min_age)

print("ages:\t\t{}".format(tf.transpose(ages)))
print("min age:\t{}".format(min_transposed1))


# *
# relaxed_mask[i][j] = MIN(i, j) >= MAX(i, j) / 1 + threshold
#
# *#



relaxed_mask = tf.math.greater_equal(tf.math.minimum(ages, tf.transpose(ages)),
                                     tf.math.divide(tf.math.maximum(ages, tf.transpose(ages)),
                                                    1 + threshold))
print("{}".format(relaxed_mask))
for i in range(7):
    for j in range(7):
        if relaxed_mask[i][j]:
            print("{} == {}".format(ages[i], ages[j]))

