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


# This implementation is based on the paper:
# Steganalysis of JPEG Images: Breaking the F5 Algorithm 
# by Jessica Fridrich, Miroslav Goljan and Dorin Hogea.
# http://www.ws.binghamton.edu/fridrich/Research/f5.pdf

# The tool "convert" from ImageMagick is needed.


import os
import sys
import shutil
import tempfile
import numpy as np
import jpeg_toolbox as jt

def H_i(dct, k, l, i):
    dct_kl = dct[k::8,l::8].flatten()
    return sum(np.abs(dct_kl) == i)

def beta_kl(dct_0, dct_b, k, l):
    h00 = H_i(dct_0, k, l, 0)
    h01 = H_i(dct_0, k, l, 1)
    h02 = H_i(dct_0, k, l, 2)
    hb0 = H_i(dct_b, k, l, 0)
    hb1 = H_i(dct_b, k, l, 1)

    return (h01*(hb0-h00) + (hb1-h01)*(h02-h01)) / (h01**2 + (h02-h01)**2)


def calibration(path, only_first_channel=False):
    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 "+path+" "+predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    shutil.rmtree(tmpdir)

    beta_list = []
    for i in range(im_jpeg["jpeg_components"]):
        dct_b = im_jpeg["coef_arrays"][i]
        dct_0 = impred_jpeg["coef_arrays"][i]
        b01 = beta_kl(dct_0, dct_b, 0, 1)   
        b10 = beta_kl(dct_0, dct_b, 1, 0)   
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01+b10+b11)/3
        beta_list.append(beta)
        if only_first_channel:
            break

    return beta_list


if __name__== "__main__": 
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<jpeg image>\n")
        sys.exit(0)

    beta_list = calibration(sys.argv[1])
    found = False
    for i in range(len(beta_list)):
        if beta_list[i]>0.01:
            print("Hidden data found in channel", i, ":", beta_list[i])
    if not found:
        print("No hidden data found")



