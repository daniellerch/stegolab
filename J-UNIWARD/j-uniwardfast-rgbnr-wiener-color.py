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
# Universal Distortion Function for Steganography in an Arbitrary Domain
# by Vojtěch Holub, Jessica Fridrich and Tomáš Denemark.

# XXX Fast implementation using Numba XXX


import os
import sys
import glob
import imageio
import scipy.signal
import numpy as np
import jpeg_toolbox as jt

from multiprocessing import Pool as ThreadPool 


np.seterr(divide = 'ignore') # XXX



import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



from numba.types import float64, int64
from numba import jit



A = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if i==0:
            A[i][j] = 0.35355339
        else:
            A[i][j] = 0.5*np.cos(np.pi*(2*j+1)*i/float(16))

@jit(nopython=True, cache=True)
def invDCT(m):
	# {{{
    return np.transpose(np.dot(np.transpose(np.dot(m,A)),A))
	# }}}


B = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if j==0:
            B[i][j] = 0.35355339
        else:
            B[i][j] = 0.5*np.cos(np.pi*(2*i+1)*j/float(16))

@jit(nopython=True, cache=True)
def DCT(m):   
	# {{{
    return np.transpose(np.dot(np.transpose(np.dot(m,B)),B))
	# }}}



@jit(nopython=True, cache=True)
def custom_round(x, decimals=0):
    # {{{
    out = np.empty_like(x)
    np.round(x, decimals, out)
    return out
    # }}}

@jit(nopython=True, cache=True)
def uncompress(coeffs, quant):
    # {{{
    spatial = np.zeros(coeffs.shape)
    for block_i in range(coeffs.shape[0]//8):
        for block_j in range(coeffs.shape[1]//8):
            dct_block = coeffs[block_i*8:block_i*8+8, block_j*8:block_j*8+8]
            spatial_block = invDCT(dct_block*quant)+128
            spatial[block_i*8:block_i*8+8, block_j*8:block_j*8+8] = spatial_block
    return spatial
    # }}}

@jit(nopython=True, cache=True)
def compress(spatial, quant):
    # {{{
    dct = np.zeros(spatial.shape)
    for block_i in range(spatial.shape[0]//8):
        for block_j in range(spatial.shape[1]//8):
            spatial_block = spatial[block_i*8:block_i*8+8, block_j*8:block_j*8+8]
            dct_block = DCT(spatial_block-128)/quant
            dct[block_i*8:block_i*8+8, block_j*8:block_j*8+8] = dct_block
    return dct
    # }}}

@jit(nopython=True, cache=True)
def YCbCr_to_RGB(Y, Cb, Cr):
    # {{{
    R = Y + 1.402 * (Cr-128)
    G = Y - 0.34414 * (Cb-128)  - 0.71414 * (Cr-128) 
    B = Y + 1.772 * (Cb-128)
    return np.stack((R, G, B), -1)
    # }}}

def ternary_entropyf(pP1, pM1):
    # {{{
    p0 = 1-pP1-pM1
    P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*np.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return np.sum(H)
    # }}}

def calc_lambda(rho_p1, rho_m1, message_length, n):
    # {{{
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (np.exp(-l3 * rho_p1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = (np.exp(-l3 * rho_m1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded 
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<300:
        lamb = l1+(l3-l1)/2
        pP1 = (np.exp(-lamb*rho_p1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        pM1 = (np.exp(-lamb*rho_m1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1;
    return lamb
    # }}}

def embedding_simulator(x, rho_p1, rho_m1, m):
    # {{{
    n = x.shape[0]*x.shape[1]
    lamb = calc_lambda(rho_p1, rho_m1, m, n)
    pChangeP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    pChangeM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    y = x.copy()
    randChange = np.random.rand(y.shape[0], y.shape[1])
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1;
    y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1;
    # }}}
    return y



@jit(nopython=True, cache=True)
def cost_fn_fast(coeffs, wavelet_impact_array, RC, pad_size):
    # {{{
    k, l = coeffs.shape
    rho = np.zeros((k, l))
    tempXi = np.zeros((3, 23, 23))
    sgm = 2**(-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = np.array(list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size)))
            sub_cols = np.array(list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size)))

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:,sub_cols]
                wav_cover_stego_diff = wavelet_impact_array[f_index, mod_row, mod_col]
                tempXi[f_index] = np.abs(wav_cover_stego_diff) / (np.abs(RC_sub)+sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)

    return rho
    # }}}

def cost_fn(coeffs, spatial, quant):
    # {{{


    hpdf = np.array([
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,  
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,  
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,  
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])        

    sign = np.array([-1 if i%2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    F = []
    F.append(np.outer(lpdf.T, hpdf))
    F.append(np.outer(hpdf.T, lpdf))
    F.append(np.outer(hpdf.T, hpdf))

    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    spatial_impact = {}
    for i in range(8):
        for j in range(8):
            test_coeffs = np.zeros((8, 8))
            test_coeffs[i, j] = 1
            spatial_impact[i, j] = invDCT(test_coeffs) * quant[i, j]


    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    #wavelet_impact = {}
    wavelet_impact_array = np.zeros((len(F), 8, 8, 23, 23))
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                #wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX
                wavelet_impact_array[f_index, i, j, :, :] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.)




    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1

    # Create reference cover wavelet coefficients (LH, HL, HH)
    pad_size = 16 # XXX
    spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')
    #print(spatial_padded.shape)

    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
        RC.append(f)
    RC = np.array(RC)

    rho = cost_fn_fast(coeffs, wavelet_impact_array, RC, pad_size)

    return rho

    # }}}


def cost_polarization(rho, coeffs, spatial, quant):
    # {{{

    wet_cost = 10**13
    rho_m1 = rho.copy()
    rho_p1 = rho.copy()

    m = 0.65


    #lMean = scipy.signal.correlate(spatial, np.ones((3,3)), 'same') / np.prod((2,2), axis=0)
    #lVar = (scipy.signal.correlate(spatial**2, np.ones((3,3)), 'same') / np.prod((3,3), axis=0)-lMean**2)
    #noise = np.mean(np.ravel(lVar), axis=0)
    #print("noise:", noise, "local mean:", lMean, "local var:", lVar)


    #precover = scipy.signal.wiener(spatial, (3,3), noise=0.1) # v1 #Podria ser parecido a una simple media
    precover = scipy.signal.wiener(spatial, (3,3)) # v2 Este debe ser el correcto
    #precover = scipy.signal.wiener(spatial, (3,3), noise=1) # v3
    coeffs_estim = compress(precover, quant)

    #os.system("convert -chop 4x4 "+path+" "+predfile)

    # polarize
    s = np.sign(coeffs_estim-coeffs)
    rho_p1[s>0] = m*rho_p1[s>0]
    rho_m1[s<0] = m*rho_m1[s<0]

    rho_p1[rho_p1>wet_cost] = wet_cost
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_p1[coeffs>1023] = wet_cost

    rho_m1[rho_m1>wet_cost] = wet_cost
    rho_m1[np.isnan(rho_m1)] = wet_cost
    rho_m1[coeffs<-1023] = wet_cost


    return rho_p1, rho_m1
    # }}}



def hide(cover_path, stego_path, payload):
    #np.set_printoptions(suppress=True)

    jpg = jt.load(cover_path)
    #I = imageio.imread(cover_path)


    Y = uncompress(jpg["coef_arrays"][0], jpg["quant_tables"][0])
    Cb = uncompress(jpg["coef_arrays"][1], jpg["quant_tables"][1])
    Cr = uncompress(jpg["coef_arrays"][2], jpg["quant_tables"][1])
    spatial_YCbCr = (Y, Cb, Cr)
    I = YCbCr_to_RGB(Y, Cb, Cr)

    for channel in range(3):
        coeffs = jpg["coef_arrays"][channel]
        spatial = I[:,:,channel]
        quant = jpg["quant_tables"][0]
        if channel>0:
            quant = jpg["quant_tables"][1]

        rho = cost_fn(coeffs, spatial, quant)

        rho_p1, rho_m1 = cost_polarization(rho, coeffs, spatial_YCbCr[channel], quant)

        nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
        stego_coeffs = embedding_simulator(coeffs, rho_p1, rho_m1, round(payload * nzAC))
        print("channel:", channel, "modifs:", np.sum(np.abs(stego_coeffs.astype("int16")-coeffs.astype("int16"))))
        jpg["coef_arrays"][channel] = stego_coeffs
        
    jt.save(jpg, stego_path)

    # }}}



if __name__ == "__main__":
 

    if len(sys.argv) == 4 and os.path.isfile(sys.argv[1]):
        payload = sys.argv[3]
        hide(sys.argv[1], sys.argv[2], float(payload))

    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[1]):
        payload = sys.argv[3]

        def worker(params): 
            i, f = params
            d = os.path.join(sys.argv[2], os.path.basename(f))
            if os.path.exists(d):
                print("Already exists:", d)
                return
            if "-" in payload:
                np.random.seed(i)
                rng = payload.split('-')
                rini = float(rng[0])
                rend = float(rng[1])
                rnd_payload = np.random.uniform(rini, rend)
                print(f, round(rnd_payload, 3))
                try:
                    hide(f, d, rnd_payload)
                except:
                    print("Cannot hide", f)
                    pass
            else:
                print(f)
                hide(f, d, float(payload))

        params = []
        files = glob.glob(sys.argv[1]+'/*.*') 
        i = 0
        for f in files:
            params.append( (i, f) )
            i += 1

        print("files:", len(files))
        pool = ThreadPool(os.cpu_count()) 
        pool.map(worker, params) 
        pool.close() 
        pool.terminate() 
        pool.join() 


    else:
        print("Usage:", sys.argv[0], "<cover file/dir> <stego file/dir> <payload>")
        sys.exit(0)


