from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
import numpy as np

def multi_model():
    x1 = Input(shape=(3,))
    x2 = Input(shape=(3,))
    y1 = Dense(10)(x1)
    y2 = Dense(10)(x2)
    y1 = Lambda(lambda x:0.1*x)(y1)
    concat = Add()([y1, y2])
    return Model([x1, x2], concat)

if __name__ == '__main__':
    model = multi_model()
    model.summary()
    x = np.ones((5, 3))
    x_ = np.zeros((5, 3))
    y = np.ones((5, 10))
    model.compile(loss='mae', optimizer='adam')
    result = model.predict([x, x_])
    print(result)