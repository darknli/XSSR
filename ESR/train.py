# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# KTF.set_session(sess)

from ESR.model import GAN
from ESR.data import dataDiscriminator, dataGenertor


epochs = 100
d_batch_size = 2
g_batch_size = 1
step_epoch = 1
crop_w = 128
crop_h = 256

if __name__ == '__main__':
    train_g = dataGenertor(r"E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4", batch_size=g_batch_size, label=1, crop_w=crop_w, crop_h=crop_h)
    val_g = dataGenertor(r'E:\Data\SR\DIV2K\DIV2K_test_LR_bicubic\X4', batch_size=g_batch_size, label=1, crop_w=crop_w, crop_h=crop_h)

    train_d = dataDiscriminator(r'E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4',
                                r'E:\Data\SR\DIV2K\DIV2K_train_HR',
                                batch_size=d_batch_size
                                )
    model = GAN(3, 32)

    for _ in range(epochs):
        model.train_discriminator(step_epoch, train_d)
        model.train_generator(step_epoch, train_g, val_g)

