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
from sklearn.preprocessing import minmax_scale

ALPHA = 2.

STATES = {
    "A": ["A", "B"],
    "B": ["C", "D"],
    "C": ["E", "F"],
    "D": ["G", "H"],
    "E": ["A", "B"],
    "F": ["C", "D"],
    "G": ["E", "F"],
    "H": ["G", "H"]
}



def get_watermark_from_password(passw, shape):
    hash = SHA256.new()
    hash.update(passw.encode('utf-8'))
    seed = int.from_bytes(hash.digest(), "little") % 2**32
    np.random.seed(seed)
    mark = np.random.randint(256, size=shape).astype('float32')
    mark -= np.mean(mark) # zero mean
    mark /= np.std(mark) # unit variance
    return mark

def E_TRELLIS_8(c_o, m_i, passw):

    state = "A"
    w_mi = []
    for i in range(len(m_i)):
        next_state = STATES[state][int(m_i[i])]
        arc = state+next_state
        w_r = get_watermark_from_password(passw+":"+arc+":"+str(i), c_o.shape)
        w_mi.append(w_r)
        state = next_state

    w_mi = np.array(w_mi)
    w_tmp = np.sum(w_mi, axis=0)
    w_m = w_tmp/w_tmp.std()
    w_a = (ALPHA*w_m)
    c_w = c_o+w_a
    c_w[c_w>255] = 255
    c_w[c_w<0] = 0
    return c_w.astype('uint8')

if __name__ == "__main__":

    if len(sys.argv)!=5 or len(sys.argv[3])!=8:
        print("Usage:\n   ", sys.argv[0], "<cover> <password> <message> <marked>\n")
        print("   cover:    Cover image to be marked")
        print("   password: Password used for generating the watermark")
        print("   message:  Bits to hide (byte) from 00000000 to 11111111)")
        print("   marked:   Name of the output (marked) image")
        print("")
        sys.exit(0)

    c_o = imread(sys.argv[1])
    m_i = sys.argv[3]
    c_w = E_TRELLIS_8(c_o, m_i+"00", sys.argv[2])
    imwrite(sys.argv[4], c_w)


