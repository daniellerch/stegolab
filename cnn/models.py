
# Common image steganalysis models adapted to color images


# {{{ XuNet()
def XuNet(input_shape=None):

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras import optimizers
    from tensorflow.keras import initializers
    from tensorflow.keras import regularizers

    if input_shape == None:
        input_shape = (512, 512, 3)

    inputs = Input(shape=input_shape)
    x = inputs

    F0 = np.array(
       [[-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1]])

    F = np.reshape(F0, (F0.shape[0],F0.shape[1],1,1) )
    bias=np.array([0])
    print(F.shape)

    w = np.concatenate((F, F, F), axis=2)
    print(w.shape)

    i0 = inputs[:,:,:,0:1]
    i1 = inputs[:,:,:,1:2]
    i2 = inputs[:,:,:,2:3]
    x0 = Conv2D(1, (5,5), padding="same", data_format="channels_last", weights=[F,bias])(i0)
    x1 = Conv2D(1, (5,5), padding="same", data_format="channels_last", weights=[F,bias])(i1)
    x2 = Conv2D(1, (5,5), padding="same", data_format="channels_last", weights=[F,bias])(i2)
    x = concatenate([x0, x1, x2], axis=3)
    print(x)

    x = Conv2D(8, (5,5), padding="same", strides=1, data_format="channels_last")(x)
    x = BatchNormalization()(x)
    x = Lambda(K.abs)(x)
    x = Activation("tanh")(x)
    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_last")(x)
    print(x)

    x = Conv2D(16, (5,5), padding="same", data_format="channels_last")(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_last")(x)
    print(x)

    x = Conv2D(32, (1,1), padding="same", data_format="channels_last")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_last")(x)
    print(x)

    x = Conv2D(64, (1,1), padding="same", data_format="channels_last")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_last")(x)
    print(x)

    x = Conv2D(128, (1,1), padding="same", data_format="channels_last")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_last")(x)
    print(x)

    x = GlobalAveragePooling2D(data_format="channels_last")(x)
    print(x)

    x = Dense(2)(x)
    x = Activation('softmax')(x)

    predictions = x

    model = Model(inputs=inputs, outputs=predictions)

    return model
# }}}

# {{{ SRNet()
def SRNet(input_shape=None):
    """
    Deep Residual Network for Steganalysis of Digital Images. M. Boroumand,
    M. Chen, J. Fridrich. http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
    """

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras import optimizers
    from tensorflow.keras import initializers
    from tensorflow.keras import regularizers



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
# }}}

# {{{ EffnetB0()
def EffnetB0(input_shape=None):

    # pip install efficientnet
    import efficientnet.tfkeras as efn

    if input_shape == None:
        input_shape = (512, 512, 3)

    model = tf.keras.Sequential([
        efn.EfficientNetB0(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
            ),
        L.GlobalAveragePooling2D(),
        L.Dense(2, activation='softmax')
        ])
    return model
# }}}




