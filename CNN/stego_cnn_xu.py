#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Network presented in the paper:
    Structural Design of Convolutional Neural Networks for Steganalysis.
    Xu, Guanshuo and  Wu, Han-Zhou and Shi, Yun-Qing. 
    IEEE Signal Processing Letters, vol. 23, issue 5, pp. 708-712. 05/2016.
"""


import sys
import glob
from scipy import misc, ndimage, signal
import numpy
import random
import ntpath
import os
from skimage.util.shape import view_as_blocks, view_as_windows

from keras.layers import Merge, Lambda, Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.core import Reshape
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# Crop squared blocks of the image with size:
n=512

F0 = numpy.array(
   [[-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]])


# {{{ load_images()
def load_images(path_pattern):

    files=glob.glob(path_pattern)

    X=[]
    for f in files:
        I = misc.imread(f)
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )


    X=numpy.array(X)

    return X
# }}}


if len(sys.argv) < 3:
   print("%s <train cover dir> <train stego dir> <test cover dir> <test stego dir>\n" % sys.argv[0])
   sys.exit(0)


Xc = load_images(sys.argv[1]+'/*.pgm')
Xs = load_images(sys.argv[2]+'/*.pgm')
Yc = load_images(sys.argv[3]+'/*.pgm')
Ys = load_images(sys.argv[4]+'/*.pgm')

X = numpy.vstack((Xc, Xs))
Y = numpy.vstack((Yc, Ys))


Xt = numpy.hstack(([0]*len(Xc), [1]*len(Xs)))
Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))


Xt = np_utils.to_categorical(Xt, 2)
Yt = np_utils.to_categorical(Yt, 2)

idx=range(len(X))
random.shuffle(idx)

X=X[idx]
Xt=Xt[idx]


model = Sequential()

F = numpy.reshape(F0, (F0.shape[0],F0.shape[1],1,1) )
bias=numpy.array([0])


print F.shape

model.add(Conv2D(1, (5,5), padding="same", data_format="channels_first", input_shape=(1,n,n), activation='relu', weights=[F, bias]))


# {{{ Group 1
model.add(Conv2D(8, (5,5), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Lambda(K.abs)) 
model.add(Activation("tanh"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 2
model.add(Conv2D(16, (5,5), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 3
model.add(Conv2D(32, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 4
model.add(Conv2D(64, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 5
model.add(Conv2D(128, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D(data_format="channels_first")) 
# }}}


model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

ep=10
i=0
while True:
    i+=ep
    model.fit(X, Xt, batch_size=64, epochs=ep, validation_data=(Y, Yt), shuffle=True)
    print "****", i, "epochs done"
    open("model_xu_"+str(i)+".json", 'w').write(model.to_json()) 
    model.save_weights("model_xu_"+str(i)+".h5")



