#!/usr/bin/env python3

import os
import sys
import glob
import imageio
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import numpy as np
import jpeg_toolbox as jt

#from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ThreadPool 


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')



def ternary_entropyf(pP1, pM1):
    p0 = 1-pP1-pM1
    P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*np.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return np.sum(H)


def calc_lambda(rho_p1, rho_m1, message_length, n):
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



def embedding_simulator(x, rho_p1, rho_m1, m):
    n = x.shape[0]*x.shape[1]
    lamb = calc_lambda(rho_p1, rho_m1, m, n)
    pChangeP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    pChangeM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    y = x.copy()
    randChange = np.random.rand(y.shape[0], y.shape[1])
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1;
    y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1;
    return y


def hide(cover_path, stego_path, payload):

    #np.set_printoptions(suppress=True)

    jpg = jt.load(cover_path)
    spatial = imageio.imread(cover_path)

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
            spatial_impact[i, j] = idct2(test_coeffs) * jpg["quant_tables"][0][i, j]

    # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    wavelet_impact = {}
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX


    # Create reference cover wavelet coefficients (LH, HL, HH)
    pad_size = 16 # XXX
    spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')
    #print(spatial_padded.shape)

    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
        RC.append(f)


    coeffs = jpg["coef_arrays"][0]
    k, l = coeffs.shape
    nzAC = np.count_nonzero(jpg["coef_arrays"][0]) - np.count_nonzero(jpg["coef_arrays"][0][::8, ::8])

    rho = np.zeros((k, l))
    tempXi = [0.]*3
    sgm = 2**(-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size))
            sub_cols = list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size))

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:,sub_cols]
                wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub)+sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)


    wet_cost = 10**13
    rho_m1 = rho.copy()
    rho_p1 = rho.copy()

    rho_p1[rho_p1>wet_cost] = wet_cost
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_p1[coeffs>1023] = wet_cost

    rho_m1[rho_m1>wet_cost] = wet_cost
    rho_m1[np.isnan(rho_m1)] = wet_cost
    rho_m1[coeffs<-1023] = wet_cost

    stego_coeffs = embedding_simulator(coeffs, rho_p1, rho_m1, round(payload * nzAC))
    #print(np.sum(np.abs(stego_coeffs.astype("int16")-coeffs.astype("int16"))))
    #print(stego_coeffs)


    jpg["coef_arrays"][0] = stego_coeffs
    jt.save(jpg, stego_path)




if __name__ == "__main__":
 
    if len(sys.argv) == 4 and os.path.isfile(sys.argv[1]):
        hide(sys.argv[1], sys.argv[2], float(sys.argv[3]))

    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[1]):

        def worker(f): 
            print(f)
            d = os.path.join(sys.argv[2], os.path.basename(f))
            hide(f, d, float(sys.argv[3]))

        files = glob.glob(sys.argv[1]+'/*.*') 
        print("files:", len(files))
        pool = ThreadPool(os.cpu_count()) 
        pool.map(worker, files) 
        pool.close() 
        pool.terminate() 
        pool.join() 


    else:
        print("Usage:", sys.argv[0], "<cover file/dir> <stego file/dir> <payload>")
        sys.exit(0)


