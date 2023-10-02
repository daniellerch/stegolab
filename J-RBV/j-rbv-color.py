#!/usr/bin/env python3


# Copyright (c) 2023 Daniel Lerch Hostalot. All rights reserved.
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
# Distortion function based on residual blocks for JPEG steganography
# by Qingde Wei, Zhaoxia Yin, Zichi Wang and Xinpeng Zhang.


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


# invDCT
A = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if i==0:
            A[i][j] = 0.35355339
        else:
            A[i][j] = 0.5*np.cos(np.pi*(2*j+1)*i/float(16))

@jit(nopython=True, cache=True)
def invDCT(m):
    return np.transpose(np.dot(np.transpose(np.dot(m,A)),A))
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
def compute_B(residuals):
    # {{{
    B = np.zeros((residuals.shape[0]//8, residuals.shape[1]//8))
    for i in range(8):
        for j in range(8):
            B[i, j] = np.sum(residuals[i*8:i*8+8, j*8:j*8+8])
    return B
    # }}}

@jit(nopython=True, cache=True)
def compute_rho(residuals, B, quant):
    # {{{
    rho = np.zeros((residuals.shape[0], residuals.shape[1]))
    for row in range(rho.shape[0]):
        for col in range(rho.shape[1]):
            m = row//8
            n = col//8
            i = row%8
            j = col%8
            P = 0
            cnt = 0
            if i>0: 
                P += B[i-1, j]
                cnt += 1
            if j>0: 
                P += B[i, j-1]
                cnt += 1
            if i<7: 
                P += B[i+1, j]
                cnt += 1
            if j<7: 
                P += B[i, j+1]
                cnt += 1

            rho[row, col] = quant[i, j] / (B[m, n] + P/cnt + 1e-5)
    return rho
    # }}}

@jit(nopython=True, cache=True)
def uncompress(coeffs, quant):
    # {{{
    spatial = np.zeros(coeffs.shape)
    for block_i in range(coeffs.shape[0]//8):
        for block_j in range(coeffs.shape[1]//8):
            dct_block = coeffs[block_i*8:block_i*8+8, block_j*8:block_j*8+8]
            spatial_block = invDCT(dct_block)*quant
            spatial[block_i*8:block_i*8+8, block_j*8:block_j*8+8] = spatial_block
    return spatial
    # }}}




def cost_fn(coeffs, spatial, quant):
    # {{{

    # Paper: Distortion function based on residual blocks for JPEG steganography


    # Step 1: Convert from DCT to Spatial Domain
    spatial = uncompress(coeffs, quant)


    # Step 2: Prepare high-pass filters (Daubechies-8)
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


    # Step 3: Apply filter and sum absolute values
    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial, F[i], mode='same', boundary='fill')
        RC.append(f)
    residuals = np.sum(np.abs(np.array(RC)), axis=0)

    # Step 4: Measure the risk
    B = compute_B(residuals)

    # Step 5: Compute the cost
    rho = compute_rho(residuals, B, quant)


    wet_cost = 10**13
    rho_m1 = rho.copy()
    rho_p1 = rho.copy()

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
    I = imageio.imread(cover_path)

    for channel in range(3):
        coeffs = jpg["coef_arrays"][channel]
        spatial = I[:,:,channel]
        quant = jpg["quant_tables"][0]
        if channel>0:
            quant = jpg["quant_tables"][1]
        rho_p1, rho_m1 = cost_fn(coeffs, spatial, quant)

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
        hide(sys.argv[1], sys.argv[2], float(payload))

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
                except Exception as e:
                    print("Cannot hide", f, str(e))
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


