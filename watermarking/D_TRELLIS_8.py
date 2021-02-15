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


def D_TRELLIS_8(c, passw):
    msg = ""

    z = { state: 0.0 for state in STATES } # costs
    p = { state: [] for state in STATES }  # paths

    p["A"] = ["A"]
    for i in range(8):
        valid_states = [s for s in p if len(p[s])]
        new_p = {}
        new_z = { s: 0.0 for s in STATES }
        for state in valid_states:
            for bit in [0, 1]:
                next_state = STATES[state][bit]
                arc = state+next_state
                w_r = get_watermark_from_password(passw+":"+arc+":"+str(i), c.shape)
                Z_lc = np.mean(c*w_r)

                # Two paths to the same node, get the lowest cost one.
                if not next_state in new_p or new_z[next_state] < z[state]+Z_lc:
                    new_z[next_state] = z[state]+Z_lc
                    new_p[next_state] = p[state]+[next_state]

        p = dict(new_p)
        z = dict(new_z)

    mx = max(z, key=z.get)
    for i in range(len(p[mx])-1):
        if STATES[p[mx][i]][0] == p[mx][i+1]:
            msg += "0"
        elif STATES[p[mx][i]][1] == p[mx][i+1]:
            msg += "1"
        else:
            print("ERROR: Wrong state!")
            sys.exit(0)

    return msg

if __name__ == "__main__":

    if len(sys.argv)!=3:
        print("Usage:\n   ", sys.argv[0], "<image> <password>\n")
        print("   image:    Image to check")
        print("   password: Password used for generating the watermark")
        print("")
        sys.exit(0)

    c = imread(sys.argv[1])
    msg = D_TRELLIS_8(c, sys.argv[2])
    print("msg:", msg)

