#!/usr/bin/python3

import os
import sys
import shutil
import tempfile
import numpy as np
import jpeg_toolbox as jt

def H_i(dct, k, l, i):
    dct_kl = dct[::k+1,::l+1].flatten()
    return sum(np.abs(dct_kl) == i)

def beta_kl(dct_0, dct_b, k, l):
    h00 = H_i(dct_0, k, l, 0)
    h01 = H_i(dct_0, k, l, 1)
    h02 = H_i(dct_0, k, l, 2)
    hb0 = H_i(dct_b, k, l, 0)
    hb1 = H_i(dct_b, k, l, 1)

    return (h01*(hb0-h00) + (hb1-h01)*(h02-h01)) / (h01**2 + (h02-h01)**2)


def calibration(path):
    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 "+path+" "+predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    found = False
    for i in range(im_jpeg["jpeg_components"]):
        dct_b = im_jpeg["coef_arrays"][i]
        dct_0 = impred_jpeg["coef_arrays"][i]
        b01 = beta_kl(dct_0, dct_b, 0, 1)   
        b10 = beta_kl(dct_0, dct_b, 1, 0)   
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01+b10+b11)/3
        if beta > 0.05:
            print("Hidden data found in channel "+str(i)+":", beta)
            found = True

    if not found:
        print("No hidden data found", beta)

    shutil.rmtree(tmpdir)



if __name__== "__main__": 
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<jpeg image>\n")
        sys.exit(0)

    calibration(sys.argv[1])



