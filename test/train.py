import numpy as np
from test.model import GAN



if __name__ == '__main__':
    gan = GAN()
    print('\n\n*****************')
    gan.generator.trainable = False
    gan.combined.summary()
    print('\n\n*****************')
    gan.generator.trainable = True
    gan.combined.summary()
    print('\n\n*****************')
    gan.discriminator.trainable = False
    gan.combined.summary()
    print('\n\n*****************')
    gan.discriminator.trainable = True
    gan.combined.summary()
    print('\n\n*****************')
    print('\n\n*****************')
    gan.generator.trainable = False
    gan.combined.summary()
    # gan.train(3000)