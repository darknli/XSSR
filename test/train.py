import numpy as np
from test.model import GAN



if __name__ == '__main__':
    gan = GAN()
    gan.train(3000)