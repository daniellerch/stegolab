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


# Matrix Embedding using binary random codes.


import sys
import random
import numpy as np


class RC:
    def __init__(self, n, k, min_value=0, max_value=255):
        # {{{
        self.code = 2 # binary
        self.n = n
        self.k = k
        self.msg_len = n-k
        self.block_len = n
        self.gen_H()
        self.max_value=max_value
        self.min_value=min_value

        # Prepare lookup table
        self.D = {}
        for i in range(self.code**n):
            new_y = np.array([int(b) for b in list(bin(i)[2:].zfill(n))])
            new_m = self.H.dot(new_y)%self.code
            if new_m.tobytes() in self.D:
                self.D[new_m.tobytes()].append(new_y)
            else:
                self.D[new_m.tobytes()] = [new_y]

        # }}}
 
    def gen_H(self):
        # {{{

        # H = [I, RND]
        H = np.identity(self.n-self.k).astype('int8')

        while H.shape[1]<self.n:
            V = np.array([ random.randint(0,1) for i in range(self.n-self.k) ])
            if np.sum(V) == 0:
                continue
            lindep = False
            for col in H:
                if np.array_equal(col, V):
                    lindep = True
                    break
            if lindep:
                continue
            H = np.append(H, V[...,np.newaxis], axis=1)

        self.H = H
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

    def _embed(self, c, m):
        # {{{

        s = c.copy()
        i=j=0
        while True:
            m_chunk = m[i:i+self.msg_len]
            c_chunk = c[j:j+self.block_len]%self.code


            if len(m_chunk) == 0:
                break

            if len(m_chunk) != self.msg_len:
                print("ERROR: wrong message length:", m_chunk)
                sys.exit(0)

            if len(c_chunk)!=self.block_len:
                print("ERROR: wrong cover length or message too long:", c_chunk)
                sys.exit(0)

            closest_idx = np.argmin(np.sum(np.array(self.D[m_chunk.tobytes()])!=c_chunk, axis=1))
            s_chunk = self.D[m_chunk.tobytes()][closest_idx]
            for k in range(self.block_len):
                if s[j:j+self.block_len][k]%self.code != s_chunk[k]:
                    add = random.choice([1, -1])
                    if s[j:j+self.block_len][k]+1 > self.max_value:
                        add = -1
                    elif s[j:j+self.block_len][k]-1 < self.min_value:
                        add = 1
                    s[j:j+self.block_len][k] += add

            i += self.msg_len
            j += self.block_len

        return s
        # }}}

    def embed(self, cover, message):
        # {{{
        shape = cover.shape
        c = cover.flatten()

        # Insert message length
        l = len(message)
        msg_len = l.to_bytes(4, 'big')

        m = self._bytes_to_bits(msg_len)
        # Left padding
        while len(m) % self.msg_len != 0:
            m = np.insert(m, 0, 0)

        c_len =(len(m)//self.msg_len)*self.block_len
        s_header = self._embed(c[:c_len], m)


        # Insert the message
        m = self._bytes_to_bits(message)

        # Right padding
        while len(m)%self.msg_len != 0:
            m = np.append(m, 0)

        s_content = self._embed(c[c_len:], m)

        s = np.append(s_header, s_content, axis=0)

        return s.reshape(shape)
        # }}}

    def _extract(self, s):
        # {{{ 
        m = []

        for i in range(0, len(s), self.block_len):
            s_chunk = s[i:i+self.block_len]%self.code
            if len(s_chunk)<self.block_len:
                break
            m_chunk = self.H.dot(s_chunk)%self.code
            m += m_chunk.tolist()

        return m
        # }}}

    def extract(self, stego):
        # {{{
        shape = stego.shape
        s = stego.flatten()

        # Read header
        l = 4*8
        while l%self.msg_len != 0:
            l += 1

        bytes_len =(l//self.msg_len)*self.block_len

        m = self._extract(s[:bytes_len])
        while (len(m))%8 != 0:
            m = m[1:]

        b = self._bits_to_bytes(m)
        msg_len = int.from_bytes(b, 'big') 

        # Read message
        m = self._extract(s[bytes_len:])
        while len(m)%8 != 0:
            m.pop()


        message_bytes = self._bits_to_bytes(m[:msg_len*8])

        return message_bytes
        # }}}


# Example
if __name__ == "__main__":

    import imageio
    cover = imageio.imread("image.png")

    print(f"Cover:\n{cover[:10,:10,0]}\n")


    message = "Hello World".encode('utf8')
    print(f"Message to hide: {message}\n")

    hc = RC(9, 5)
    print(f"Shared matrix H:\n{hc.H}\n")

    stego = cover.copy()
    stego[:,:,0] = hc.embed(cover[:,:,0], message)
    print(f"Stego:\n{stego[:10,:10,0]}\n")

    imageio.imsave("stego.png", stego)

    stego = imageio.imread("stego.png")
    extracted_message = hc.extract(stego[:,:,0])
    print("Extracted message:", extracted_message.decode())
    





