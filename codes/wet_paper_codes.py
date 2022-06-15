#!/usr/bin/env python

# Copyright (c) 2022 Daniel Lerch Hostalot. All rights reserved.
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


# Wet Paper Codes using binary Hamming codes.


import sys
import random
import itertools
import numpy as np


class WPC:
    def __init__(self, msg_len, min_value=0, max_value=255):
        # {{{
        self.code = 2 # binary
        self.msg_len = msg_len
        self.block_len = self.code**msg_len-1
        self.gen_H()
        self.max_value=max_value
        self.min_value=min_value
        # }}}
 
    def _n_ary(self, n):
        # {{{
        if n == 0:
            return '0'
        nums = []
        while n:
            n, r = divmod(n, self.code)
            nums.append(str(r))
        return ''.join(reversed(nums))
        # }}}

    def gen_H(self):
        # {{{
        H = []
        l = len(self._n_ary(self.block_len))
        for i in range(1, self.code**self.msg_len):
            string = self._n_ary(i).zfill(l)
            V=[int(c) for c in string]
            H.append(V)
        self.H=np.array(H).T
        # }}}

    def _bytes_to_bits(self, m):
        # {{{
        bits=[]
        for b in m:
            for i in range(8):
                bits.append((b >> i) & 1)
        return np.array(bits)
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

    def _find_ij(self, H, i):
        # {{{
        for i2 in range(i, H.shape[1]):
            sm = 0
            j2 = -1
            for j in range(i+0, H.shape[0]):
                sm += H[j, i2]
                if H[j, i2] == 1:
                    j2 = j
            if sm == 1:
                return i2, j2
        return -1, -1
        # }}}

    def _back_substitution(self, H, z):
        # {{{
        m = H.shape[0]
        v = [0]*m

        for i in range(m-1, -1, -1):
            v[i] = z[i]
            for j in range(i+1, m):
                v[i] -= H[i][j] * v[j]

        return np.array(v)%2
        # }}}

    def _matrix_lt_process(self, A, b):
        # {{{
        H = A.copy()
        z = b.copy()

        # Solve Hv = z
        i = 0
        m = H.shape[0]
        tau = []
        while i<m:
            i2, j2 = self._find_ij(H, i)
            if i2<0 or j2<0:
                return [] # Fail

            if i!=j2:
                # swap rows
                H[i], H[j2] = H[j2].copy(), H[i].copy() 
                z[i], z[j2] = z[j2].copy(), z[i].copy()

            if i!=i2:
                # swap columns
                H[:,i], H[:,i2] = H[:,i2].copy(), H[:,i].copy()
                tau.append( (i, i2) ) # save swapped columns

            i = i + 1

        v = self._back_substitution(H, z)
        v = np.array(v.tolist() + [0]*(H.shape[1]-len(v)))

        # Undo column swap
        for i, i2 in reversed(tau):
            v[i], v[i2] = v[i2], v[i]

        return v
        # }}}

    def embed(self, cover, wet, message):
        # {{{
        shape = cover.shape
        c = cover.flatten()
        w = wet.flatten()
        m = self._bytes_to_bits(message)

        s = c.copy()
        i=j=0
        while True:
            m_chunk = m[i:i+self.msg_len]
            c_chunk = c[j:j+self.block_len]%self.code
            w_chunk = w[j:j+self.block_len]%self.code
            w_indices = np.where(w_chunk==1)[0]

            if len(m_chunk) == 0:
                break

            if len(m_chunk) > 0 and len(c_chunk)!=self.block_len:
                print("WARNING: message too long")
                break

            # padding with zeros
            # if len(m_chunk)!=self.msg_len:
            #    m_chunk = np.array(m_chunk.tolist() + [0]*(self.msg_len-len(m_chunk)))

            # remove wet values
            H = np.delete(self.H, w_indices[::-1], 1) 

            z = (m_chunk-self.H.dot(c[j:j+self.block_len]%self.code))%2
            v = self._matrix_lt_process(H, z)

            if len(v)==0:
                print("ERROR: Too many wet pixels: ", w_chunk)
                sys.exit(0)


            # insert zeros in wet positions
            for idx in w_indices:
                v = np.insert(v, idx, 0)

            for k in range(len(v)):
                if v[k] == 1:
                    add = random.choice([1, -1])
                    if v[k] >= self.max_value:
                        add = -1
                    elif v[k] <= self.min_value:
                        add = 1
                    s[j:j+self.block_len][k] += add

            i += self.msg_len
            j += self.block_len

        return s.reshape(shape)
        # }}}

    def extract(self, stego):
        # {{{ 
        shape = cover.shape
        s = stego.flatten()
        m = []

        for i in range(0, len(s), self.block_len):
            s_chunk = s[i:i+self.block_len]%self.code
            if len(s_chunk)<self.block_len:
                break
            m_chunk = self.H.dot(s_chunk)%self.code
            #print("s_chunk:", s_chunk)
            #print("m_chunk:", m_chunk)
            m += m_chunk.tolist()

        message_bytes = self._bits_to_bytes(m)
        return message_bytes
        # }}}


# Example
if __name__ == "__main__":

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


    wet = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 1, 1],
    ])


    print(f"Cover:\n{cover}\n")
    print(f"Wet:\n{wet}\n")

    message = "HELO".encode('utf8')
    print(f"Message to hide: {message}\n")

    wpc = WPC(3)
    print(f"Shared matrix H:\n{wpc.H}\n")

    stego = wpc.embed(cover, wet, message)
    print(f"Stego:\n{stego}\n")

    print("Modifications:")
    print(np.abs(cover-stego))

    extracted_message = wpc.extract(stego)
    print("Extracted message:", extracted_message.decode())






