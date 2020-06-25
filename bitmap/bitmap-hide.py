#!/usr/bin/python3

import imageio


# Read image
img = imageio.imread("lena.png")

# Modify the coefficient in position (6,6) from channel 0
img[6,6,0] += 1

# Save modified image
imageio.imwrite("lena_stego.png", img)


