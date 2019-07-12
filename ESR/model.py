from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from functools import partial


class ResidualDenseBlock_5c:
    def __init__(self, nf=64, gc=32, bias=True):
        self.model = self.get_model(nf, gc, bias)

    def get_model(self, nf, gc, bias):
        input_x = Input(None, None, 3)
        concat = []
        x = input_x
        for i in range(4):
            x = Conv2D(nf + i*gc, gc, (3, 3), (1, 1), padding="same", bias=bias, activation=LeakyReLU(0.2))(x)
            concat.append(x)
            x = Concatenate(axis=-1)(concat)
        x = Conv2D(nf+5*gc, gc, (3, 3), (1, 1), padding="same", bias=bias)(x)
        output = Add()(x*0.2 + x)
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class RRDB:
    def __init__(self, nf, gc=32):
        self.model = self.get_model(nf, gc)

    def get_model(self, nf, gc):
        input_x = Input(None, None, 3)
        RDB1 = ResidualDenseBlock_5c(nf, gc)
        RDB2 = ResidualDenseBlock_5c(nf, gc)
        RDB3 = ResidualDenseBlock_5c(nf, gc)
        
        x = RDB1(input_x)
        x = RDB2(x)
        x = RDB3(x)
        
        output = Add()([x*0.2, x])
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class RRDBNet:
    def __init__( self, in_nc ,out_nc, nf, nb=23, gc=32):
        self.model = self.get_model(in_nc ,out_nc, nf, nb, gc)

    def get_model(self, in_nc ,out_nc, nf, nb, gc):
        RRBD_block_f = partial(RRDB, nf=nf, gc=gc)
        input_x = Input(None, None, 3)
        fea = Conv2D(in_nc, nf, (3, 3), (1, 1), padding="same")(input_x)
        x = fea
        for i in range(nb):
            x = RRBD_block_f(x)
        x = Add()([x, fea])

        x = UpSampling2D((2, 2))(x)
        x = LeakyReLU(0.2)(x)
        x = UpSampling2D((2, 2))(x)
        x = LeakyReLU(0.2)(x)
        output = Conv2D(nf, out_nc, (3, 3), padding="same")(x)
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class ImageNet:
    def __init__(self, model_name, weights="imagenet"):
        self.model = self.get_model(model_name, weights)

    def get_model(self, model_name, weights):
        if model_name == 'InceptionV3':
            from tensorflow.python.keras.applications.inception_v3 import InceptionV3
            base_model = InceptionV3(weights=weights, include_top=False)
        elif model_name == 'NASNetLarge':
            from tensorflow.python.keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(weights=weights, include_top=False)
        elif model_name == 'DenseNet201':
            from tensorflow.python.keras.applications.densenet import DenseNet201
            base_model = DenseNet201(weights=weights, include_top=False)
        elif model_name == 'Xception':
            from tensorflow.python.keras.applications.xception import Xception
            base_model = Xception(weights=weights, include_top=False)
        elif model_name == 'VGG19':
            from tensorflow.python.keras.applications.vgg19 import VGG19
            base_model = VGG19(weights=weights, include_top=False)
        elif model_name == 'NASNetMobile':
            from tensorflow.python.keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(weights=weights, include_top=False)
        elif model_name == 'MobileNetV2':
            from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(weights=weights, include_top=False)
        elif model_name == 'ResNet50':
            from tensorflow.python.keras.applications.resnet50 import ResNet50
            base_model = ResNet50(weights=weights, include_top=False)
        elif model_name == 'InceptionResNetV2':
            from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
            base_model = InceptionResNetV2(weights=weights, include_top=False, )

        else:
            raise KeyError('Unknown network.')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class GAN:
    def __init__(self, in_nc ,out_nc, nf, nb=23, gc=32, discriminator_name='VGG16', discriminator_weights=None):
        self.generator = RRDBNet(in_nc ,out_nc, nf, nb=23, gc=32)
        self.discriminator = ImageNet(discriminator_name, discriminator_weights)
        self.gan = self.get_model()

    def get_model(self):
        input_x = Input(None, None, 3)
        x = self.generator(input_x)
        output = self.discriminator(x)
        model = Model(input_x, output)
        return model

    def train_generator(self, epochs, train, val):
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.gan.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            validation_data=val,
            validation_steps=len(val),
            epochs=epochs,
            workers=8,
            use_multiprocessing=True,
            max_queue_size=100,
        )

    def train_discriminator(self, epochs, train, val):
        self.discriminator.trainable = True
        self.generator.trainable = False
        self.gan.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            validation_data=val,
            validation_steps=len(val),
            epochs=epochs,
            workers=8,
            use_multiprocessing=True,
            max_queue_size=100,
        )
    