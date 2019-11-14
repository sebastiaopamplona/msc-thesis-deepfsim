import datetime

import tensorflow as tf


class NotifyWhileAway(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        checkpoint = "Epoch {}/{} - tra_loss: {}  |  val_loss: {}".format(epoch+1,
                                                                          self.total_epochs,
                                                                          round(float(logs["loss"]), 4),
                                                                          round(float(logs["val_loss"]), 4))
        print(checkpoint)
        f = open("C:/Users/Sebastião Pamplona/Desktop/to_sync_w_google_drive/training_iter_1.txt", "a+")
        f.write("{}\n".format(checkpoint))
        f.close()
