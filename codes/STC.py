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
# Minimizing Embedding Impact in Steganography using Trellis-Coded Quantization
# by Tomas Filler, Jan Judas, and Jessica Fridrich.

# Part of this code is based on the Matlab implementation provided here: 
# http://dde.binghamton.edu/download/syndrome/

# For a Python interface to the efficient C++ implementation go to:
# https://github.com/daniellerch/pySTC


import sys
import random
import numpy as np


class STC:
    def __init__(self, stcode, k):
        # {{{
        n_bits = len(bin(np.max(stcode))[2:])

        M = []
        for d in stcode:
            M.append(np.array([int (x) for x in list(bin(d)[2:].ljust(n_bits, '0'))]))
        M = np.array(M).T
        H_hat = M
        n, m = H_hat.shape

        H = np.zeros((k+n-1, m*k))
        for i in range(k):
            H[i:i+n, m*i:m*(i+1)] = H_hat;

        self.code_n = m*k
        self.code_l = n_bits
        self.code_h = np.tile(stcode, k)
        self.code_shift = np.tile([0]*(m-1)+[1], k)
        # }}}
    
    def _dual_viterbi(self, x, w, m):
        # {{{ 

        C = np.zeros((2**self.code_l, self.code_n))
        costs = np.infty * np.ones((2**self.code_l, 1))
        costs[0] = 0
        paths = np.zeros((2**self.code_l, self.code_n))

        m_id = 0 # message bit id
        y = np.zeros(x.shape)

        # Run forward
        for i in range(self.code_n):
            costs_old = costs.copy()
            hi = self.code_h[i]
            ji = 0
            for j in range(2**self.code_l):
                c1 = costs_old[ji] + x[i]*w[i]
                c2 = costs_old[(ji^hi)] + (1-x[i])*w[i]
                if c1<c2:
                    costs[j] = c1
                    paths[j, i] = ji # store index of the previous path
                else:
                    costs[j] = c2
                    paths[j, i] = ji^hi # store index of the previous path
                ji = ji + 1

            for j in range(self.code_shift[i]):
                tail = np.infty* np.ones((2**(self.code_l-1),1)) 
                if m[m_id] == 0:
                    costs = np.vstack((costs[::2], tail))
                else:
                    costs = np.vstack((costs[1::2], tail))
                    
                m_id = m_id + 1

            C[:,i] = costs[:,0]

        # Backward run
        ind = np.argmin(costs)
        min_cost = costs[ind,0]

        m_id -= 1

        for i in range(self.code_n-1, -1, -1):
            for j in range(self.code_shift[i]):
                ind = 2*ind + m[m_id,0]  # invert the shift in syndrome trellis
                m_id = m_id - 1

            y[i] = paths[ind, i]!=ind
            ind = int(paths[ind, i])

        return y.astype('uint8'), min_cost, paths
    # }}}

    def _calc_syndrome(self, x):
        # {{{
        m = np.zeros((np.sum(self.code_shift), 1))
        m_id = 0
        tmp = 0
        for i in range(self.code_n):
            hi = self.code_h[i]
            if x[i] == 1:
                tmp = hi ^ tmp
            for j in range(self.code_shift[i]):
                m[m_id] = tmp%2
                #tmp = tmp >> 1
                tmp //= 2
                m_id += 1
        return m.astype('uint8')
        # }}}

    def _bytes_to_bits(self, m):
        # {{{
        bits=[]
        for b in m:
            for i in range(8):
                bits.append((b >> i) & 1)
        return bits
        # }}}
    
    def _bits_to_bytes(self, m):
        # {{{
        enc = bytearray()
        idx=0
        bitidx=0
        bitval=0
        for b in m:
            if bitidx==8:
                enc.append(bitval)
                bitidx=0
                bitval=0
            bitval |= b<<bitidx
            bitidx+=1
        if bitidx==8:
            enc.append(bitval)

        return bytes(enc)
        # }}}

    def embed(self, cover, costs, message):
        # {{{
        shape = cover.shape
        x = cover.flatten()
        w = costs.flatten()
        ml = np.sum(self.code_shift)
        message_bits = np.array(self._bytes_to_bits(message))

        i = 0
        j = 0
        y = x.copy()
        while True:
            x_chunk = x[i:i+self.code_n][:,np.newaxis]%2
            m_chunk = message_bits[j:j+ml][:,np.newaxis]
            w_chunk = w[i:i+self.code_n][:,np.newaxis]
            y_chunk, min_cost, _ = self._dual_viterbi(x_chunk, w_chunk, m_chunk)
            idx = x_chunk[:,0] != y_chunk[:,0]
            y[i:i+self.code_n][idx] += 1
            i += self.code_n
            j += ml
            if i+self.code_n>len(x) or j+ml>len(message_bits):
                break
        return np.vstack(y).reshape(shape)
        # }}}

    def extract(self, stego):
        # {{{ extract()
        shape = cover.shape
        y = stego.flatten()
        message = []
        for i in range(0, len(y), self.code_n):
            y_chunk = y[i:i+self.code_n][:,np.newaxis]%2
            if len(y_chunk)<self.code_n:
                break
            m_chunk = self._calc_syndrome(y_chunk)
            message += m_chunk[:,0].tolist()
    
        message_bytes = self._bits_to_bytes(message)
        return message_bytes
        # }}}


cover = np.array([
    [200, 200, 201, 210, 251, 251, 251, 251],
    [200, 200, 201, 240, 239, 239, 239, 239],
    [201, 201, 201, 219, 234, 234, 234, 234],
    [201, 201, 202, 210, 205, 205, 205, 205],
    [202, 202, 210, 218, 215, 215, 215, 215],
    [202, 202, 210, 218, 215, 215, 215, 215],
    [203, 203, 213, 218, 228, 220, 235, 254],
    [203, 203, 214, 219, 220, 231, 245, 255],
])

costs = np.array([
    [100, 100,  99,  50,  50,  50,  50,  50],
    [100, 100,  99,  50,  40,  40,  40,  40],
    [100, 100,  99,  60,  50,  50,  50,  50],
    [100, 100,  95,  60,  50,  60,  55,  55],
    [100, 100,  80,  70,  70,  70,  70,  75],
    [100, 100,  80,  70,  70,  70,  70,  75],
    [100, 100,  80,  70,  70,  70,  70,  75],
    [100, 100,  60,  60,  65,  60,  55, 999],
])


print("costs:\n", costs)
print("cover:\n", cover)

message = "HELO".encode('utf8')
print("Message to hide:", message)

stcode = [71, 109] # See Table 1 for other good syndrome-trellis codes.
stc = STC(stcode, 8) 
print("cover len:", stc.code_n)
print("message len:", np.sum(stc.code_shift))

stego = stc.embed(cover, costs, message)
print("stego:\n", stego)


extracted_message = stc.extract(stego)
print("Extracted message:", extracted_message.decode())




