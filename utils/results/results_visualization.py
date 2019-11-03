import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.constants import WIKI_ALIGNED_160_ABS

cmap = "cool"
def get_query_object(string):
    return string.split("(")[1].split("=")[1].split(",")[0]


def get_neighbor(string):
    return string.split(",")[1]


def create_knn_images(criterion, triplet_strategy, embeddings_cnn,  model_name):
    experiments_path = "../../experiments/{}/{}/".format(criterion, triplet_strategy)
    results_path = "{}rlc_results/{}/{}/".format(experiments_path,
                                                 embeddings_cnn,
                                                 model_name)
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass

    Nr = 2
    Nc = 6
    with open("{}knn_results.txt".format(results_path)) as f:
        for k in range(22):
            images = []
            fig, axs = plt.subplots(Nr, Nc)

            for i in range(Nr):
                for j in range(Nc):
                    axs[i, j].axis("off")

            # <age>|<idx>.png
            query = get_query_object(f.readline())
            fig.suptitle('KNN\n{}, k={}'.format(query, Nc))
            img = mpimg.imread("{}{}".format(WIKI_ALIGNED_160_ABS, query.split("|")[1]))
            print(img)
            images.append(axs[0, 0].imshow(img, cmap=cmap))
            axs[0, 0].set_title(query.replace("|", "\n"))

            for i in range(Nc):
                # <age>|<idx>.png
                neighbor = get_neighbor(f.readline())
                neighbor = neighbor[0:len(neighbor) - 1]
                img = mpimg.imread("{}{}".format(WIKI_ALIGNED_160_ABS, neighbor.split("|")[1]))
                images.append(axs[1, i].imshow(img, cmap=cmap))
                axs[1, i].set_title(neighbor.replace("|", "\n"))

            plt.savefig("{}knn_{}".format(results_path, query.split("|")[1]))


create_knn_images(criterion="age",
                  triplet_strategy="adapted_semihard",
                  embeddings_cnn="vgg16",
                  model_name="es_128_e_5_bs_32_ts_27910_s_1_as_1_ar_1_ai_0_u_0_fa_1")

# path = 'C:\\Users\\Sebastião Pamplona\\Desktop\\DEV\\datasets\\mtcnn_extracted\\'
# with open("..\\experiments\\age_knn_k10_rlc_tests_UNFROZEN_m1.txt") as f:
#     for k in range(22):
#         images = []
#         fig, axs = plt.subplots(Nr, Nc)
#
#         for i in range(Nr):
#             for j in range(Nc):
#                 axs[i, j].axis("off")
#
#         # <age>|<idx>.png
#         query = get_query_object(f.readline())
#         fig.suptitle('KNN\n{}, k=5'.format(query))
#         print("{}{}".format(path, query.split("|")[1]))
#         img = mpimg.imread("{}{}".format(path, query.split("|")[1]))
#         images.append(axs[0, 0].imshow(img, cmap=cmap))
#         axs[0, 0].set_title(query.replace("|", "\n"))
#
#         for i in range(5):
#             # <age>|<idx>.png
#             neighbor = get_neighbor(f.readline())
#             neighbor = neighbor[0:len(neighbor)-1]
#             print("{}{}".format(path, neighbor.split("|")[1]))
#             img = mpimg.imread("{}{}".format(path, neighbor.split("|")[1]))
#             images.append(axs[1, i].imshow(img, cmap=cmap))
#             axs[1, i].set_title(neighbor.replace("|", "\n"))
#
#         plt.savefig("knn_{}".format(query.split("|")[1]))



# for k in range(2):
#     images = []
#     fig, axs = plt.subplots(Nr, Nc)
#     fig.suptitle('Multiple images')
#     path = 'C:\\Users\\Sebastião Pamplona\\Desktop\\DEV\\datasets\\mtcnn_extracted\\8294.png'
#     img = mpimg.imread(path)
#     images.append(axs[0, 0].imshow(img, cmap=cmap))
#
#     for i in range(Nr):
#         for j in range(Nc):
#             axs[i, j].axis("off")
#
#     # axs[0, 0].label_outer()
#     for j in range(Nc):
#         # Generate data with a range that varies from one plot to the next.
#
#
#         img = mpimg.imread(path)
#         images.append(axs[1, j].imshow(img, cmap=cmap))
#         axs[1, j].axis('off')
#         axs[1, j].label_outer()
#
#     plt.axis('off')
#     plt.savefig("{}.png".format(k))

# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
# def update(changed_image):
#     for im in images:
#         if (changed_image.get_cmap() != im.get_cmap()
#                 or changed_image.get_clim() != im.get_clim()):
#             im.set_cmap(changed_image.get_cmap())
#             im.set_clim(changed_image.get_clim())
#
#
# for im in images:
#     im.callbacksSM.connect('changed', update)


