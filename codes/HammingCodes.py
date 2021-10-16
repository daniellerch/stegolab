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


# Matrix Embedding using binary Hamming codes.


import sys
import random
import numpy as np


class HC:
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

    def embed(self, cover, message):
        # {{{
        shape = cover.shape
        c = cover.flatten()
        m = self._bytes_to_bits(message)
        print(m)

        s = c.copy()
        i=j=0
        while True:
            m_chunk = m[i:i+self.msg_len]
            c_chunk = c[j:j+self.block_len]%self.code

            if len(m_chunk) == 0:
                break

            if len(m_chunk) > 0 and len(c_chunk)!=self.block_len:
                print("WARNING: message too long")
                break

            # padding with zeros
            # if len(m_chunk)!=self.msg_len:
            #    m_chunk = np.array(m_chunk.tolist() + [0]*(self.msg_len-len(m_chunk)))

            column = (self.H.dot(c_chunk)-m_chunk)%self.code

            # Find position of column in H
            position = np.where(np.sum(np.abs(self.H.T-column), axis=1)==0)[0]

            if len(position)>0:
                v = s[j:j+self.block_len][position[0]]
                add = random.choice([1, -1])
                if v == self.max_value:
                    add = -1
                elif v == self.min_value:
                    add = 1
                s[j:j+self.block_len][position[0]] += add

            i += self.msg_len
            j += self.block_len

            #if i+self.msg_len>len(m) or j+self.block_len>len(c):
            #    break

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

    print(f"Cover:\n{cover}\n")

    message = "HELO".encode('utf8')
    print(f"Message to hide: {message}\n")

    hc = HC(3)
    print(f"Shared matrix H:\n{hc.H}\n")

    stego = hc.embed(cover, message)
    print(f"Stego:\n{stego}\n")

    extracted_message = hc.extract(stego)
    print("Extracted message:", extracted_message.decode())






