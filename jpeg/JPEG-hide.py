#!/usr/bin/python3

import sys
import random
import numpy as np

# https://github.com/daniellerch/python-jpeg-toolbox
import jpeg_toolbox as jt

# Read image
img = jt.load("lena.jpg")

# Modify the coefficient in position (6,6) from channel 0
img["coef_arrays"][0][6,6] += 1

# Save modified image
jt.save(img, "lena_stego.jpg")


