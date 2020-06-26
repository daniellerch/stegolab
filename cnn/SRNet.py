
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers                                                                  

# Deep Residual Network for Steganalysis of Digital Images. M. Boroumand, 
# M. Chen, J. Fridrich. http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf

def create_model(input_shape=None):

    if input_shape == None:
        input_shape = (512, 512, 3)

    inputs = Input(shape=input_shape)
    x = inputs

    conv2d_params = {
        'padding': 'same',
        'data_format': 'channels_last',
        'bias_initializer': initializers.Constant(0.2),
        'bias_regularizer': None,
        'kernel_initializer': initializers.VarianceScaling(),
        'kernel_regularizer': regularizers.l2(2e-4),
    }

    avgpool_params = {
        'padding': 'same',
        'data_format': 'channels_last',
        'pool_size': (3,3),
        'strides': (2,2)
    }

    bn_params = {
        'momentum': 0.9,
        'center': True,
        'scale': True
    }


    x = Conv2D(64, (3,3), strides=1, **conv2d_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = Activation("relu")(x)

    x = Conv2D(16, (3,3), strides=1, **conv2d_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = Activation("relu")(x)

    for i in range(5):
        y = x
        x = Conv2D(16, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation("relu")(x)
        x = Conv2D(16, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = add([x, y])
        y = x


    for f in [16, 64, 128, 256]:
        y = Conv2D(f, (1,1), strides=2, **conv2d_params)(x)
        y = BatchNormalization(**bn_params)(y)
        x = Conv2D(f, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation("relu")(x)
        x = Conv2D(f, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = AveragePooling2D(**avgpool_params)(x)
        x = add([x, y])

    x = Conv2D(512, (3,3), **conv2d_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3,3), **conv2d_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = GlobalAveragePooling2D(data_format="channels_first")(x)

    x = Dense(2, kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01),
                 bias_initializer=initializers.Constant(0.) )(x)
    x = Activation('softmax')(x)

    predictions = x

    model = Model(inputs=inputs, outputs=predictions)

    return model



