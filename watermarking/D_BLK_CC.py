#!/usr/bin/env python3

# Copyright (c) 2020 Daniel Lerch Hostalot. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

# Implementation of a watermarking system described in the book:
# Digital Watermarking and Steganography. Morgan Kaufmann. 2007.
# by I. J. Cox, M. L. Miller, J. A. Bloom, J. Fridrich and T. Kalker.



import sys
import numpy as np
from imageio import imread, imwrite
from Crypto.Hash import SHA256

T_CC = 0.55

def get_watermark_from_password(passw, shape):
    hash = SHA256.new()
    hash.update(passw.encode('utf-8'))
    seed = int.from_bytes(hash.digest(), "little") % 2**32
    np.random.seed(seed)
    mark = np.random.randint(256, size=shape).astype('float32')
    mark -= np.mean(mark) # zero mean
    mark /= np.std(mark) # unit variance
    return mark

def D_BLK_CC(c, w_r):

    v = np.zeros((8,8)).astype('float32')
    for i in range(8):
        for j in range(8):
            v[i, j] = np.mean(c[i::8,j::8])

    v -= v.mean()
    w_r -= w_r.mean()

    return np.sum(v*w_r) / np.sqrt(np.sum(v*v)*np.sum(w_r*w_r))

if __name__ == "__main__":

    if len(sys.argv)!=3:
        print("Usage:\n   ", sys.argv[0], "<image> <password>\n")
        print("   image:    Image to check")
        print("   password: Password used for generating the watermark")
        print("")
        sys.exit(0)

    c = imread(sys.argv[1])
    w_r = get_watermark_from_password(sys.argv[2], (8,8))
    Z_cc = D_BLK_CC(c, w_r)
    print("CC:", Z_cc)

    if Z_cc > T_CC:
        print("watermark, m=1")
    elif Z_cc < -T_CC:
        print("watermark, m=0")
    else:
        print("no watermark")



