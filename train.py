import io
import os

from keras.callbacks.callbacks import ModelCheckpoint

from utils.loss_functions.semihard_triplet_loss import adapted_semihard_triplet_loss
from utils.models.callbacks import NotifyWhileAway
from utils.models.models import create_concatenated_model
from utils.utils import get_args, get_argsDG, get_labels, get_tra_val_tes_size, get_dgs, \
    get_in_out_labels, \
    get_path_checkpoints, get_path_embeddings, \
    get_optimizers_dict, get_parameters_details

if __name__ == '__main__':
    args = get_args()

    # Create the model
    model = create_concatenated_model(args)
    model.compile(loss=adapted_semihard_triplet_loss,
                  optimizer=get_optimizers_dict(learning_rate=args.learning_rate)[args.optimizer])
    # model.summary()

    # Configuring the DataGenerator for the training, validation and test set
    argsDG = get_argsDG(args)
    if args.age_interval == 0:
        labels = get_labels(args)
        tra_sz, val_sz, tes_sz = get_tra_val_tes_size(set_size=len(labels),
                                                      split_train_val=90,
                                                      split_train_test=90)
        tra_dg, val_dg, tes_dg = get_dgs(labels=labels,
                                         tra_sz=tra_sz,
                                         val_sz=val_sz,
                                         tes_sz=tes_sz,
                                         args=args,
                                         argsDG=argsDG)
        sz = len(labels)
    elif args.age_interval == 1:
        in_labels, in_sz, out_labels, out_sz = get_in_out_labels(args)

        # tra_sz, val_sz, tes_sz = get_tra_val_tes_size(set_size=in_sz,
        #                                               split_train_val=90,
        #                                               split_train_test=100)

        tra_sz = 3234 * len(in_labels)   # TODO: hardcoded!
        val_sz = 264 * len(in_labels)    # TODO: hardcoded!
        tes_sz = out_sz
        tra_dg, val_dg, tes_dg = get_dgs(labels=None,
                                         labels_in=in_labels,
                                         labels_out=out_labels,
                                         tra_sz=tra_sz,
                                         val_sz=val_sz,
                                         args=args,
                                         argsDG=argsDG)

        sz = len(in_labels) * len(in_labels[0]) + len(out_labels)
    else:
        raise Exception('Interval {} not supported OR Criterion {} not supported'
                        .format(args.age_interval, args.criterion))

    experiments_path, model_path, model_name, details = get_parameters_details(args,
                                                                               sz,
                                                                               tra_sz,
                                                                               val_sz,
                                                                               tes_sz)
    print(details)
    f = open("C:/Users/Sebastião Pamplona/Desktop/to_sync_w_google_drive/training_iter_1.txt", "a+")
    for i in range(3):
        f.write("{}\n".format("=" * 80))
    f.write("{}\n".format(details))
    f.close()

    # Train the model
    if args.train:
        print("Training ...")
        # checkpoints_path = get_path_checkpoints(model_path, model_name)
        checkpoints_path = "E:/SebastiaoPamplona/training_iter_1/{}/".format(args.optimizer)
        try:
            os.mkdir(model_path + model_name.split(".")[0])
        except FileExistsError:
            pass
        try:
            os.mkdir(checkpoints_path)
        except FileExistsError:
            pass


        # Create callback to save weights after each epoch
        weights_callback = ModelCheckpoint(checkpoints_path + "weights_e_{epoch:02d}_tra-loss_{loss:.4f}_val-loss_{val_loss:.4f}.h5",
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=False,
                                           save_weights_only=True,
                                           mode='auto',
                                           period=1)

        # Train the model
        model.fit_generator(generator=tra_dg,
                            steps_per_epoch=int(tra_sz / args.batch_size) - 1,
                            validation_data=val_dg,
                            epochs=args.num_epochs,
                            max_queue_size=1,
                            verbose=1,
                            callbacks=[weights_callback, NotifyWhileAway(args.num_epochs)])

        # Save the weights
        model.save_weights(model_path + model_name)

    # Load weights from the trained model and produce embeddings for the test set
    else:
        print("Predicting with the weights from: {} ...".format(model_path + model_name))
        embeddings_path = get_path_embeddings(experiments_path,
                                              args.embeddings_cnn,
                                              model_name)

        try:
            os.mkdir(embeddings_path)
        except FileExistsError:
            pass
        model.load_weights(model_path + model_name)

        # Produce embeddings
        e = model.predict_generator(generator=tes_dg,
                                    steps=int(tes_sz / args.batch_size) - 1,
                                    max_queue_size=1,
                                    verbose=1)
        embeddings = e[:, 1:]
        labels = e[:, :1]

        # m, s = calc_mean_and_std(embeddings)
        # print("(not normalized) mean: {} | std: {}".format(m, s))

        # Create the two necessary .tsv files for the Tensorboard visualization
        out_embeddings = io.open("{}embeddings.tsv".format(embeddings_path),
                                 'w',
                                 encoding='utf-8')
        out_labels = io.open("{}labels.tsv".format(embeddings_path),
                             'w',
                             encoding='utf-8')
        for i in range(embeddings.shape[0]):
            out_embeddings.write('{}\n'.format('\t'.join([str(x) for x in embeddings[i][:]])))
            out_labels.write('{}|{}.png\n'.format(int(labels[i][0]), i))












    # if args.criterion == "age":
    #     tra_dg = AgeDG(ages=labels[0:tra_sz],
    #                    set_size=tra_sz,
    #                    **argsDG)
    #     val_dg = AgeDG(ages=labels[tra_sz:tra_sz + val_sz],
    #                    set_size=val_sz,
    #                    **argsDG)
    #     tes_dg = AgeDG(ages=labels[tra_sz + val_sz:],
    #                    set_size=tes_sz,
    #                    **argsDG)
    # elif args.criterion == "eigenvalues":
    #     tra_dg = EigenvaluesDG(eigenvalues=labels[0:tra_sz],
    #                            set_size=tra_sz,
    #                            **argsDG)
    #     val_dg = EigenvaluesDG(eigenvalues=labels[tra_sz:tra_sz + val_sz],
    #                            set_size=val_sz,
    #                            **argsDG)
    #     tes_dg = EigenvaluesDG(eigenvalues=labels[tra_sz + val_sz:],
    #                            set_size=tes_sz,
    #                            **argsDG)
    # else:
    #     raise Exception('Criterion {} not supported'.format(args.criterion))

    # Configure paths
    #  = "experiments/{}/{}/{}/".format(args.dataset,
    #                                                   args.criterion,
    #                                                   args.triplet_strategy)
    #  = "{}models/{}/".format(experiments_path, args.embeddings_cnn)
    #  = get_model_name(args, tra_sz)


'''
    data_gen_params = {'batch_size': args.batch_size,
                       'dim': (args.image_size, args.image_size, args.n_image_channels),
                       'embedding_size': args.embedding_size,
                       'dataset_path': args.dataset_path,
                       'img_format': dataset_to_img_format[args.dataset]}

    ages = get_ages(args.age_interval, args.dataset_path)
    set_size = len(ages)

    tra_size, val_size, tes_size = get_tra_val_tes_size(set_size=set_size,
                                                        split_train_val=95,
                                                        split_train_test=95)

    train_generator = AgeDG(ages=ages[0:tra_size], set_size=tra_size, **data_gen_params)
    validation_generator = AgeDG(ages=ages[tra_size:tra_size + val_size], set_size=val_size, **data_gen_params)
    test_generator = AgeDG(ages=ages[tra_size + val_size:], set_size=tes_size, **data_gen_params)

    data_generator_params = {'batch_size': 66,
                             'dim': (160, 160, 3),
                             'embedding_size': 128}
                             # 'dataset_path': args.dataset_path,
                             # 'img_format': dataset_to_img_format[args.dataset]}

    tra_relaxed_ages = pickle.load(
        open("{}in\\relaxed_ages.pickle".format(args.dataset_path), 'rb'))
    tes_relaxed_ages = pickle.load(
        open("{}out\\relaxed_ages.pickle".format(args.dataset_path), 'rb'))
    tra_size = len(tra_relaxed_ages)
    tes_size = len(tes_relaxed_ages)

    train_generator = AgeRelaxedIntervalDG(relaxed_ages=tra_relaxed_ages,
                                           set_size=tra_size,
                                           training_flag=1,
                                           **data_generator_params)

    test_generator = AgeRelaxedIntervalDG(relaxed_ages=tes_relaxed_ages,
                                          set_size=tes_size,
                                          training_flag=0,
                                          **data_generator_params)


    experiments_path = "experiments/{}/{}/{}/".format(args.dataset, args.criterion, args.triplet_strategy)
    model_path = "{}models/{}/".format(experiments_path, args.embeddings_cnn)
    model_name = get_model_name(args, tra_size)
    print(model_path + model_name)


    print_fit_details(args, set_size, tra_size, val_size, tes_size)

    print_fit_details(args, tra_size + tes_size, tra_size, 0, tes_size)
    '''


# e = model.predict_generator(generator=train_generator,
        #                             steps=int(tra_size / args.batch_size) - 1,
        #                             # callbacks=None,
        #                             max_queue_size=1,
        #                             # workers=1,
        #                             verbose=1)


        # e = model.predict_generator(generator=test_generator,
        #                             steps=int(tes_size / args.batch_size) - 1,
        #                             # callbacks=None,
        #                             max_queue_size=1,
        #                             # workers=1,
        #                             verbose=1)

        # tg = AgeDG(ages=ages[:5000], set_size=5000, **data_gen_params)
