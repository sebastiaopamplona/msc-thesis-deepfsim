from PIL import Image
import random
import os
import time
import dlib
import cv2
import multiprocessing as mp

from scipy.io import loadmat
from datetime import datetime
from imutils.face_utils import FaceAligner

from utils.constants import WIKI_ALIGNED_224_ABS, WIKI_CROP_PATH_ABS, IMDB_CROP_PATH_ABS, DESKTOP_PATH_ABS
from utils.utils import to_pickle

IMDB_CROP_PATH_REL = "..\\..\\..\\..\\datasets\\raw\\imdb_crop\\"
IMDB_ALIGNED_PATH_REL = "..\\..\\..\\..\\datasets\\treated\\age\\imdb_aligned\\"
DESKTOP_PATH_REL = "..\\..\\..\\..\\..\\"

shape_predictor = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
fa = FaceAligner(predictor, desiredFaceWidth=224)


def in_age_scope(age, min_age, max_age):
    """
    Checks if an age is inside the age scope [min_age, max_age].
    """
    return min_age <= age <= max_age


def conv_to_windows_path(path):
    """
    Replaces "/" with "\\".
    """
    return path.replace("/", "\\")


def get_age_from_filename(filename):
    """
    Extracts the age from a filename.
    Filename format:
    <age>_<?>_<?>_<?>__<?>_<dob (yyyy-mm-dd)>_<taken (yyyy)>.png
    (eg. 65_5_4519_69__10988169_1944-02-17_2009.png)
    """
    return int(filename.split("_")[0])


def show_face_image_from_image(face_image, title):
    """
    Shows an image, given an image array.
    """
    cv2.imshow(title, face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_face_image_from_path(image_path, title):
    """
    Shows an image, given an image path.
    """
    image_path = WIKI_CROP_PATH_ABS + conv_to_windows_path(image_path)
    face_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def align_face_image(image_path):
    """
    Aligns a face, with the line connecting the eyes horizontally.

    :param image_path: relative path to the face image

    :return: aligned face image
    """
    # image_path = WIKI_CROP_PATH + conv_to_windows_path(image_path)
    # image_path = IMDB_CROP_PATH + conv_to_windows_path(image_path)
    face_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    return fa.align(face_image, gray, rects[0])


def calc_age(taken, dob):
    """
    Calculates the age of a person, based on the date the photo was taken
    and on the date of birth.

    :param taken: date the photo was taken
    :param dob: date of birth

    :return: age
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def calc_age_imdb(filename):
    """
    Calculates face age, based on file name
    Filename format:
    <?>_<?>_<dob (yyyy-mm-dd)>_<taken (yyyy)>.jpg
    (eg. nm0000100_rm12818688_1955-1-6_2003.jpg)
    :param filename: file name

    :return: age
    """

    taken = int(filename.split("_")[-1].split(".")[0])
    dob = filename.split("_")[2]
    dob_year = int(dob.split("-")[0])
    dob_month = int(dob.split("-")[1])

    # assume the photo was taken in the middle of the year
    if dob_month < 7:
        return taken - dob_year
    else:
        return taken - dob_year - 1


def get_meta(mat_path, db):
    """
    Parses the metadata file into a dictionary.

    :param mat_path: path to the metadata file
    :param db: dataset (eg. wiki, imdb)

    :return: a dictionary containing "file_name", "gender", "age",
     "score" and "second_score"
    """
    # This is hard coded because of a zlib bug
    meta = loadmat("C:\\Users\\Sebastião Pamplona\\Downloads\\wiki.mat")
    # meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
            "second_score": second_face_score}

    return data


def get_meta_chunk(meta, start_index, end_index):
    """
    Returns the metadata corresponding to the chunk of images.

    :param meta: the whole metadata
    :param start_index: the start index
    :param end_index: the end index

    :return: metadata chunk
    """
    return {"file_name": meta["file_name"][start_index:end_index + 1],
            "gender": meta["gender"][start_index:end_index + 1],
            "age": meta["age"][start_index:end_index + 1],
            "score": meta["score"][start_index:end_index + 1],
            "second_score": meta["second_score"][start_index:end_index + 1]}


def contains_face(score):
    """
    If the image score is -inf, it means it does not contain a face.

    :param score: face image score

    :return: bool
    """
    return score != (-float('Inf'))


def treat_chunk_imdb(pid, dirs):
    """
    Aligns the faces of a chunk of images.
    Computes the age of a face image, based on the filename.
    Deletes corrupted images.
    The IMDB dataset is divided in 100 folders, named "00" to
    "99"; each chunk treat a certain range (eg. 00/* to 24/*).
    Filename format:
    <?>_<?>_<dob>_<taken>.jpg
    (eg. nm0000100_rm12818688_1955-1-6_2003.jpg)

    :param pid: process id
    :param dirs: chunk dirs range

    :return: number of face images aligned, and removed
    """

    print("[{}] Treating dirs {} to {}...".format(pid, dirs[0], dirs[len(dirs) - 1]))
    ok = 0
    removed = 0
    ctr = 0
    for d in dirs:
        print("[{}] Current: {}".format(pid, d))
        imgs = os.listdir("{}{}".format(IMDB_CROP_PATH_ABS, d))
        for img in imgs:
            img = d + "\\" + img
            try:
                print("[{}] {}".format(pid, ctr))
                img_aligned = align_face_image("{}{}".format(IMDB_CROP_PATH_REL, img))
                # print(img_aligned)
                h, w, c = img_aligned.shape
                if c != 3:
                    removed += 1
                else:
                    # cv2.imwrite("{}{}_{}.jpg".format(DESKTOP_PATH, ctr, calc_age_imdb(img)),
                    #             img_aligned)
                    filename_name = "{}{}\\{}_{}.jpg".format(IMDB_ALIGNED_PATH_REL,
                                                           d,
                                                           ctr,
                                                           calc_age_imdb(img))
                    cv2.imwrite(filename_name, img_aligned)
                    ok += 1
                    ctr += 1
            except Exception as e:
                print(e)
                print("\t{} as no face".format(img))
                removed += 1

    print("[{}] OK: {} REMOVED: {}...".format(pid, ok, removed))


def treat_chunk(pid, meta, start_index, end_index, min_age, max_age):
    """
    Aligns the faces of a chunk of images, removing images that:
    1. reside outside of the age scope [<min_age>, <max_age>]
    2. do not contain a face
    3. are corrupted
    Filename format:
    <age>_<pid>_<for loop index>_<original file name>.png

    :param pid: process id
    :param meta: metadata chunk
    :param start_index: start index
    :param end_index: end index
    :param min_age: minimum age for the age scope
    :param max_age: maximum age for the age scope

    :return: number of successfully parsed, number of removed
    """

    ok = 0
    removed = 0
    chunk_size = end_index - start_index + 1
    print("[{}] Treating {} images ({}-{})...".format(pid, chunk_size, start_index, end_index))

    for index in range(0, end_index - start_index + 1):
        if in_age_scope(meta["age"][index], min_age, max_age) and contains_face(meta["score"][index]):
            img_path = conv_to_windows_path(meta["file_name"][index][0])
            ptr = img_path.split("\\")[0] + "__" + img_path.split("\\")[1].split(".")[0]
            try:
                img_aligned = align_face_image(img_path)
                h, w, c = img_aligned.shape
                if c != 3:
                    removed += 1
                else:
                    cv2.imwrite("{}{}_{}_{}_{}.png".format(WIKI_ALIGNED_224_ABS,
                                                           meta["age"][index],
                                                           pid, index, ptr),
                                img_aligned)
                    ok += 1
            except Exception as e:
                removed += 1
        else:
            removed += 1

    print('[{}] DONE'.format(pid))
    print('[{}]\t\tTreated: {} images'.format(pid, ok + removed))
    print('[{}]\t\tAligned: {} images'.format(pid, ok))
    print('[{}]\t\tRemoved: {} images'.format(pid, removed))

    return ok, removed


def treat_age_dataset_multi_CPU(db, min_age, max_age):
    """
    Treats the age dataset parallelly, excluding people outside of the
    age scope [<min_age>, <max_age>].

    :param db: dataset (eg. wiki, imdb)
    :param min_age: minimum age for the age scope
    :param max_age: maximum age for the age scope
    """

    meta = get_meta("{}.mat".format(db), db)
    size = len(meta["file_name"])
    num_cpu = mp.cpu_count()
    block_size = size // num_cpu
    rest_block_size = size % num_cpu + block_size
    processes = []
    if size % num_cpu == 0:
        assert block_size * num_cpu == size
        for i in range(num_cpu):
            start_index = i * block_size
            end_index = (i + 1) * block_size - 1

            processes.append(mp.Process(target=treat_chunk, args=(i,
                                                                  get_meta_chunk(meta, start_index, end_index),
                                                                  start_index,
                                                                  end_index,
                                                                  min_age,
                                                                  max_age)))

    else:
        assert block_size * (num_cpu - 1) + rest_block_size == size
        for i in range(num_cpu - 1):
            start_index = i * block_size
            end_index = (i + 1) * block_size - 1

            processes.append(mp.Process(target=treat_chunk, args=(i,
                                                                  get_meta_chunk(meta, start_index, end_index),
                                                                  start_index,
                                                                  end_index,
                                                                  min_age,
                                                                  max_age)))

        start_index = size - rest_block_size
        end_index = size - 1

        processes.append(mp.Process(target=treat_chunk, args=(num_cpu - 1,
                                                              get_meta_chunk(meta, start_index, end_index),
                                                              start_index,
                                                              end_index,
                                                              min_age,
                                                              max_age)))

    start = int(round(time.time() * 1000))
    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    end = int(round(time.time() * 1000))
    print('Time elapsed: {} ms'.format(end - start))


def test_no_copies(idx1, idx2):
    """
    Compares the filepaths of two files, <idx1>.txt and <idx2>.txt,
    assertinr there are no duplicates.
    To use it, create a txt with the filenames when treating each
    dataset chunk, with the function treat_chunk.

    :param idx1: <idx1>.txt
    :param idx2: <idx2>.txt
    """

    f1 = open("{}.txt".format(idx1), "r")
    content1 = f1.read()
    f2 = open("{}.txt".format(idx2), "r")
    content2 = f2.read()
    print("Testing {}-{}...".format(idx1, idx2), end="")
    for line1 in content1.split("\n")[:-1]:
        for line2 in content2.split("\n")[:-1]:
            assert line1 != line2

    print(" OK")


def rename_chunk(pid, dataset_path, files_chunk, start_index, end_index):
    """
    Renames a chunk of images that were treated. These images have filenames that are not
    consistent with the ages array, but have the age present.
    So, a new ages array is created, with the right connection between the filename and
    the ages array (eg. 243.png is ages[234] years old).
    """
    chunk_size = end_index - start_index + 1
    print("[{}] Extracting ages from {} images ({}-{})...".format(pid, chunk_size, start_index, end_index))
    ages = []
    ctr = start_index
    for filename in files_chunk:
        age = get_age_from_filename(filename)
        ages.append(age)
        os.rename("{}{}".format(dataset_path, filename),
                  "{}{}.png".format(dataset_path, ctr))
        # print("[{}] {}".format(pid, ctr))
        ctr += 1

    print(ages)
    to_pickle(ages, "ages_{}_to_{}.pickle".format(start_index, end_index))


def rename_aligned_images_multi_CPU(dataset_path):
    """
    Executes function rename_chunk parallelly.
    """
    files = os.listdir(dataset_path)
    size = len(files)
    num_cpu = mp.cpu_count()
    block_size = size // num_cpu
    rest_block_size = size % num_cpu + block_size
    processes = []
    if size % num_cpu == 0:
        assert block_size * num_cpu == size
        for i in range(num_cpu):
            start_index = i * block_size
            end_index = (i + 1) * block_size - 1

            processes.append(mp.Process(target=rename_chunk, args=(i, dataset_path,
                                                                   files[start_index:end_index + 1],
                                                                   start_index, end_index)))

    else:
        assert block_size * (num_cpu - 1) + rest_block_size == size
        for i in range(num_cpu - 1):
            start_index = i * block_size
            end_index = (i + 1) * block_size - 1

            processes.append(mp.Process(target=rename_chunk, args=(i, dataset_path,
                                                                   files[start_index:end_index + 1],
                                                                   start_index, end_index)))

        start_index = size - rest_block_size
        end_index = size - 1

        processes.append(mp.Process(target=rename_chunk, args=(num_cpu - 1, dataset_path,
                                                               files[start_index:end_index + 1],
                                                               start_index, end_index)))

    start = int(round(time.time() * 1000))
    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    end = int(round(time.time() * 1000))
    print('Time elapsed: {} ms'.format(end - start))


def get_age_distribution(ages):
    """
    Calculates the distribution of ages.
    """
    distr = {}
    for age in ages:
        try:
            ctr = distr[age]
            distr[age] = ctr + 1
        except Exception as e:
            distr[age] = 1

    ks = list(distr.keys())
    for k in ks:
        distr[k] = round(distr[k] / len(ages) * 100, 2)

    return distr


def treat_age_dataset_IMDB_multi_CPU():
    """
    Executes function treat_chunk_imdb parallelly.
    """
    dirs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]
    for d in range(10, 100):
        dirs.append(str(d))

    size = len(dirs)
    num_cpu = mp.cpu_count()
    block_size = size // num_cpu
    rest_block_size = size % num_cpu + block_size
    processes = []
    if size % num_cpu == 0:
        assert block_size * num_cpu == size
        for i in range(num_cpu):
            start_index = i * block_size
            end_index = (i + 1) * block_size
            processes.append(mp.Process(target=treat_chunk_imdb, args=(i, dirs[start_index:end_index])))

    else:
        assert block_size * (num_cpu - 1) + rest_block_size == size
        for i in range(num_cpu - 1):
            start_index = i * block_size
            end_index = (i + 1) * block_size
            processes.append(mp.Process(target=treat_chunk_imdb, args=(i, dirs[start_index:end_index])))

        start_index = size - rest_block_size
        processes.append(mp.Process(target=treat_chunk_imdb, args=(num_cpu - 1, dirs[start_index:])))

    start = int(round(time.time() * 1000))
    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    end = int(round(time.time() * 1000))
    print('Time elapsed: {} ms'.format(end - start))
    '''
    [0] OK: 55299 REMOVED: 21301...
    [1] OK: 52753 REMOVED: 20037...
    [2] OK: 54734 REMOVED: 20813...
    [3] OK: 53401 REMOVED: 20120...
    [4] OK: 54172 REMOVED: 20379...
    [5] OK: 63785 REMOVED: 23929...
    Time elapsed: 15613525 ms
    '''


if __name__ == '__main__':
    treat_age_dataset_IMDB_multi_CPU()
    # for i in range(10):
    #     os.mkdir("{}0{}".format(IMDB_ALIGNED_PATH_REL, i))
    #
    # for i in range(10, 100):
    #     os.mkdir("{}{}".format(IMDB_ALIGNED_PATH_REL, i))
    # treat_age_dataset_multi_CPU(db="wiki", min_age=18, max_age=65)
    # rename_aligned_images_multi_CPU(WIKI_ALIGNED_PATH)

    # ages_all = []
    #
    # ages_0_to_5742 = from_pickle("ages_0_to_5742.pickle")
    # ages_all += ages_0_to_5742
    #
    # ages_5743_to_11485 = from_pickle("ages_5743_to_11485.pickle")
    # ages_all += ages_5743_to_11485
    #
    # ages_11486_to_17228 = from_pickle("ages_11486_to_17228.pickle")
    # ages_all += ages_11486_to_17228
    #
    # ages_17229_to_22971 = from_pickle("ages_17229_to_22971.pickle")
    # ages_all += ages_17229_to_22971
    #
    # ages_22972_to_28714 = from_pickle("ages_22972_to_28714.pickle")
    # ages_all += ages_22972_to_28714
    #
    # ages_28715_to_34457 = from_pickle("ages_28715_to_34457.pickle")
    # ages_all += ages_28715_to_34457
    #
    # print(len(ages_all))
    # print(ages_all)
    #
    # to_pickle(ages_all, "ages.pickle")

    # files = os.listdir(WIKI_ALIGNED_PATH)
    # assert len(files) == len(ages_all)
    # for i in range(len(files)):
    #     curr_age = get_age_from_filename(files[i])
    #     assert curr_age == ages_all[i]

    # ages = from_pickle("ages.pickle")
    # print(len(ages))
    # print(get_age_distribution(ages))
    #
    # print(ages[27910-1])

    # ages_28715_to_34457 = from_pickle("ages_28715_to_34457.pickle")
    # print(ages_28715_to_34457[-1])

    # random.sample(population, k)

    # a = []
    # for i in range(34458):
    #     a.append(i)
    #
    # subset_idx = random.sample(a, 27910)
    # subset_age = []
    # for idx in subset_idx:
    #     subset_age.append(ages[idx])
    #
    # subset_age_dist = get_age_distribution(subset_age)
    #
    # keys = subset_age_dist.keys()
    # keys = sorted(keys)
    # dic = {}
    # for key in keys:
    #     dic[key] = subset_age_dist[key]
    #
    # print(dic)
    # idxs = []
    # for i in range(30):
    #     idxs.append(i)
    #
    # idxs = set(idxs)
    # train = random.sample(idxs, 20)
    # for i in train:
    #     idxs.remove(i)
    #
    # valid = random.sample(idxs, 5)
    # for i in valid:
    #     idxs.remove(i)
    #
    # test = list(idxs)
    #
    # print(train)
    # print(valid)
    # print(test)

    # ages_dict = {}
    # for i in range(len(ages)):
    #     ages_dict[i] = ages[i]
    #
    # min_diff = 999999
    # final_age_shuffle = []
    # final_ages_shuffled_meta = []
    # for j in range(400):
    #     ages_shuffled = []
    #     ages_shuffled_meta = []
    #     keys = list(ages_dict.keys())
    #     random.shuffle(keys)
    #
    #     # print(ages_dict)
    #     # print(ages)
    #     for key in keys:
    #         ages_shuffled.append(ages_dict[key])
    #         ages_shuffled_meta.append("{}_{}".format(key, ages_dict[key]))
    #     # print(ages_shuffled)
    #     # print(ages_shuffled_meta)
    #
    #     ages_dist = get_age_distribution(ages)
    #     ages_shuffled_dist = get_age_distribution(ages_shuffled[:27910])
    #
    #     ks = list(ages_dist.keys())
    #     diff = 0
    #     for k in ks:
    #         diff += abs(ages_dist[k] - ages_shuffled_dist[k])
    #         # print("{} vs {}".format(ages_dist[k], ages_shuffled_dist[k]))
    #
    #     if diff < min_diff:
    #         min_diff = diff
    #         final_age_shuffle = ages_shuffled
    #         final_ages_shuffled_meta = ages_shuffled_meta
    #         print("New min diff: {}".format(diff))
    #
    # ages_dist = get_age_distribution(ages)
    # ages_shuffled_dist = get_age_distribution(final_age_shuffle[:27910])
    #
    # ks = list(ages_dist.keys())
    # diff = 0
    # for k in ks:
    #     diff += abs(ages_dist[k] - ages_shuffled_dist[k])
    #
    #
    # to_pickle(final_age_shuffle, "ages_shuffled.pickle")
    # to_pickle(final_ages_shuffled_meta, "ages_shuffled_meta.pickle")
    #
    # print("Final diff: {}".format(diff))

    # ages = from_pickle("ages.pickle")
    # ages_shuffled = from_pickle("ages_shuffled.pickle")
    # ages_shuffled_meta = from_pickle("ages_shuffled_meta.pickle")
    #
    # print(ages)
    # print(ages_shuffled)
    # print(ages_shuffled_meta)
    # path = "..\\..\\..\\..\\datasets\\treated\\age\\wiki_aligned_COPY_TESTING\\"
    # for file in os.listdir(path):
    #     # <new_idx>_<old_idx>.png
    #     old_idx = int(file.split("_")[1].split(".")[0])
    #     new_idx = int(file.split("_")[0])
    #     assert ages[old_idx] == ages_shuffled[new_idx]
    #     os.rename("{}{}".format(path, file),
    #               "{}{}.png".format(path, new_idx))
    #
    # print("ages[old_idx] == ages_shuffled[new_idx] OK")

    # for i in range(len(ages_shuffled_meta)):
    #     curr = ages_shuffled_meta[i].split("_")[0]
    #     os.rename("{}{}.png".format(path, curr),
    #               "{}{}_{}.png".format(path, i, curr))