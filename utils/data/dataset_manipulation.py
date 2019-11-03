
















"""
This file is not being used anymore.
To delete before "deployment".
"""




























import numpy
import pickle
import os
import matplotlib.image as mpimg

from utils.constants import MTCNN_EXTRACTED_PATH_ABS, MTCNN_EXTRACTED_CORRUPTED_PATH_ABS, MTCNN_EXTRACTED_OUT_AGE_SCOPE_PATH_ABS


def to_pickle(obj, filepath):
    pickle_out = open(filepath, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def reverse(path):
    for filename in os.listdir(path):
        os.rename("{}{}".format(path, filename),
                  "{}{}".format(path, filename.split("_")[1]))


def get_idx(filename):
    return int(filename.split(".")[0])


def parse_age_dataset(path, min_age, max_age):
    """
    Removes people with corrupted age (eg. -49, 255) and
    people outside the age scope (outside [min_age, max_age])
    """
    original_ages = pickle.load(open(path, 'rb'))
    different_ages_in_scope = {-1}
    corrupted_ages = []
    out_age_scope_ages = []
    ok_ages = []

    for k in range(len(original_ages)):
        # Corrupted
        if original_ages[k] < 0 or original_ages[k] > 110:
            os.rename("{}{}.png".format(MTCNN_EXTRACTED_PATH_ABS, k),
                      "{}{}_{}.png".format(MTCNN_EXTRACTED_CORRUPTED_PATH_ABS, len(corrupted_ages), k))
            corrupted_ages.append(original_ages[k])
            # to_delete_from_original.append(k)

        # Outside of the age scope ([18, 60])
        elif original_ages[k] < min_age or original_ages[k] > max_age:
            os.rename("{}{}.png".format(MTCNN_EXTRACTED_PATH_ABS, k),
                      "{}{}_{}.png".format(MTCNN_EXTRACTED_OUT_AGE_SCOPE_PATH_ABS, len(out_age_scope_ages), k))
            out_age_scope_ages.append(original_ages[k])

        # Inside of the age scope ([18, 60])
        else:
            ok_ages.append(original_ages[k])
            different_ages_in_scope.add(original_ages[k])


    print("Corrupted: {}".format(len(corrupted_ages)))
    print("Out of scope: {}".format(len(out_age_scope_ages)))
    print("Ok: {}".format(len(ok_ages)))

    to_pickle(ok_ages, "{}{}.pickle".format(MTCNN_EXTRACTED_PATH_ABS, "ages_ok"))
    to_pickle(corrupted_ages, "{}{}.pickle".format(MTCNN_EXTRACTED_CORRUPTED_PATH_ABS, "ages_corrupted"))
    to_pickle(out_age_scope_ages, "{}{}.pickle".format(MTCNN_EXTRACTED_OUT_AGE_SCOPE_PATH_ABS, "ages_out_of_scope"))


def count_ages(ages):
    age_counter = {}
    for age in ages:
        try:
            ctr = age_counter[age]
            age_counter[age] = ctr + 1
        except Exception:
            age_counter[age] = 1


    age_counter_array = []
    for age in range(110):
        try:
            print("{}: {}".format(age, age_counter[age]))
            age_counter_array.append(age_counter[age])
        except Exception:
            pass

    print(age_counter_array)


def remember_prev_idxs(files):
    prev_idxs = []
    for idx in files:
        prev_idxs.append(idx)

    to_pickle(prev_idxs, "{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "prev_idxs.pickle"))


def rename(path):
    """
    Renames the image files in order for the ith element in the ages array
    to correspond to the ith.png.
    eg.:
        before parse:
            age[3] -> 3.png
        during parse:
            3.png was corrupted / out of age scope
            3.png is removed
            [..., 2.png, 4.png, ...]
            ages_ok[3] shoul point to 3.png, but 3.png is gone (it should be
            4.png)
            so, 4.png is renamed to 3.png
        after parse:
            ages_ok[3] -> 3.png

    """
    files = os.listdir(path)
    # Rename the files with the idx only, in order to sort later
    for file in files:
        os.rename("{}{}".format(MTCNN_EXTRACTED_PATH_ABS, file),
                  "{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "{}".format(file.split(".")[0])))

    # Turn the strings into ints
    for i in range(len(files)):
        files[i] = int(files[i])

    files.sort()

    # Remember previous idx, in order to be able to test the new dataset,
    # comparing the pairs age|original_image_idx with age|new_image_idx
    remember_prev_idxs(files)

    curr = files[0]
    os.rename("{}{}".format(path, "0"),
              "{}{}".format(path, "0.png"))

    # Delete the gap between filenames
    # (eg. [..., 37.png, 40.png, ...] -> [..., 37.png, 38.png, ...]
    for i in range(1, len(files)):
        if files[i] > (curr + 1):
            os.rename("{}{}".format(path, files[i]),
                      "{}{}".format(path, "{}.png".format(curr + 1)))
            curr += 1
        else:
            os.rename("{}{}".format(path, files[i]),
                      "{}{}".format(path, "{}.png".format(files[i])))
            curr = get_idx("{}.png".format(files[i]))

    test_new_age_dataset()


def test_new_age_dataset():
    """
    This method tests the new dataset, where every image in the original dataset that's
    present in the new dataset should should have the same age as in the original.
    """

    print("Testing the new dataset... ", end="")
    prev_idxs = pickle.load(open("{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "prev_idxs.pickle"), 'rb'))
    ages_ok = pickle.load(open("{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "ages_ok.pickle"), 'rb'))
    original_ages = pickle.load(open("{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "ages.pickle"), 'rb'))

    print("prev_idxs: {}".format(len(prev_idxs)))
    print("ages_ok: {}".format(len(ages_ok)))
    print("original_ages: {}".format(len(original_ages)))

    # Compare age|original_image_idx with age|new_image_idx
    for i in range(len(prev_idxs)):
        original_age = original_ages[prev_idxs[i]]
        original_image = mpimg.imread("{}{}".format(MTCNN_EXTRACTED_COPY_PATH, "{}.png".format(prev_idxs[i])))

        new_age = ages_ok[i]
        new_image = mpimg.imread("{}{}".format(MTCNN_EXTRACTED_PATH_ABS, "{}.png".format(i)))

        assert original_age == new_age
        numpy.testing.assert_array_equal(original_image, new_image)

    print("OK")
