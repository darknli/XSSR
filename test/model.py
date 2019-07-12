from tensorflow.python.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Input, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
import numpy as np
import cv2

class GAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.summary()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # from tensorflow.examples.tutorials.mnist import input_data
        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #
        # x_train, y_train = mnist.train.images, mnist.train.labels
        # x_test, y_test = mnist.test.images, mnist.test.labels
        # X_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 127.5 - 1
        # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
        # Rescale -1 to 1

        from keras.datasets import mnist
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)
            if epoch > 80 and epoch % 40:
                for i in range(8):
                    cv2.imwrite("imgs/%d_%d.jpg"%(epoch, i), ((gen_imgs[i]+1)*127.5).astype(np.uint8))
            # Train the discriminator

            for layer in self.discriminator.layers:
                layer.trainable = True
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer='adam',
                                       metrics=['accuracy'])
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            for layer in self.discriminator.layers:
                layer.trainable = False
            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)

            self.combined.compile(loss='binary_crossentropy', optimizer='adam')
            g_loss = self.combined.train_on_batch(noise, valid)
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)