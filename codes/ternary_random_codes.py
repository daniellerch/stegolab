#!/usr/bin/env python3

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


# Matrix Embedding using ternary random codes.




import sys
import random
import numpy as np


class RC3:

    def __init__(self, n, k, min_value=0, max_value=255):
        # {{{
        self.code = 3 # ternary
        self.n = n
        self.k = k
        self.msg_len = n-k
        self.block_len = n
        self._prepare_H()
        self.max_value=max_value
        self.min_value=min_value

        # Prepare lookup table
        self.D = {}
        for i in range(self.code**n):
            new_y = np.array([int(b) for b in list(np.base_repr(i, base=3).zfill(n))])
            new_m = self.H.dot(new_y)%self.code
            if new_m.tobytes() in self.D:
                self.D[new_m.tobytes()].append(new_y)
            else:
                self.D[new_m.tobytes()] = [new_y]

        # }}}
 
    def _prepare_H(self):
        # {{{
        # H = [I, RND]
        H = np.identity(self.n-self.k).astype('int8')

        while H.shape[1]<self.n:
            V = np.array([ random.randint(0,2) for i in range(self.n-self.k) ])
            if np.sum(V) == 0:
                continue
            lindep = False
            for col in H:
                if (np.array_equal(1*col, V) or np.array_equal(2*col, V) ):
                    lindep = True
                    break
            if lindep:
                continue
            H = np.append(H, V[...,np.newaxis], axis=1)

        self.H = H
        # }}}

    def _bytes_to_ternary(self, m):
        # {{{
        binary_string = ""
        for b in m:
            for i in range(8):
                binary_string += str((b >> i) & 1)

        # Message as a big integer
        bigint_m = int(binary_string, 2)

        ternary_string = np.base_repr(bigint_m, base=3)
        ternary = [int(c) for c in ternary_string ]

        return np.array(ternary)
        # }}}
    
    def _ternary_to_bytes(self, m):
        # {{{

        ternary_string = "".join([ str(c) for c in m])

        bigint_m = int(ternary_string, 3)
        binary_string = np.base_repr(bigint_m, base=2)
        m_list = [int(i) for i in binary_string]

        while len(m_list)%8 != 0:
            m_list.insert(0, 0)

        # make bytes from bits
        enc = bytearray()
        idx=0
        bitidx=0
        bitval=0
        for b in m_list:
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


            closest_idx = np.argmin(np.sum(np.abs(np.array(self.D[m_chunk.tobytes()]-c_chunk)), axis=1))
            s_chunk = self.D[m_chunk.tobytes()][closest_idx]
            for k in range(self.block_len):
                d = s_chunk[k] - s[j:j+self.block_len][k]%self.code
                add = d
                if d!=0:
                    if s[j:j+self.block_len][k]+d > self.max_value:
                        if   d == 1: add = -2
                        elif d == 2: add = -1
                    elif s[j:j+self.block_len][k]+d < self.min_value:
                        if   d ==-1: add = 2
                        elif d ==-2: add = 1
                s[j:j+self.block_len][k] += add

            i += self.msg_len
            j += self.block_len

        return s
        # }}}

    def embed(self, cover, message):
        # {{{
        shape = cover.shape
        c = cover.flatten()
        message_ternary = self._bytes_to_ternary(message)
        self._ternary_to_bytes(message_ternary)

        # -- Header --

        # Get the number of ternary symbols needed to hide the header
        max_header_value = 2**32-1
        total_header_bytes = max_header_value.to_bytes(4, 'big')
        total_header_bytes_ternary = self._bytes_to_ternary(total_header_bytes)
        total_header_length = len(total_header_bytes_ternary)

        # Encode the message length in ternary
        l = len(message_ternary)
        msg_len = l.to_bytes(4, 'big')
        m = self._bytes_to_ternary(msg_len)

        # Left padding
        while len(m) < total_header_length:
            m = np.insert(m, 0, 0)

        while len(m) % self.msg_len != 0:
            m = np.insert(m, 0, 0)

        header_length =(len(m)//self.msg_len)*self.block_len
        s_header = self._embed(c[:header_length], m)


        # -- Message --

        # Right padding
        while len(message_ternary)%self.msg_len != 0:
            message_ternary = np.append(message_ternary, 0)

        s_content = self._embed(c[header_length:], message_ternary)


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
        msg_len = (2**32-1).to_bytes(4, 'big')
        l = len(self._bytes_to_ternary(msg_len))
        while l%self.msg_len != 0:
            l += 1

        header_length =(l//self.msg_len)*self.block_len

        m = self._extract(s[:header_length])

        b = self._ternary_to_bytes(m)
        msg_len = int.from_bytes(b, 'big')

        # Read message
        m = self._extract(s[header_length:])

        m = np.array(m)
        m = m[:msg_len]

        message_bytes = self._ternary_to_bytes(m)

        return message_bytes
        # }}}



# Example
if __name__ == "__main__":

    import imageio
    cover = imageio.imread("image.png")

    print(f"Cover:\n{cover[:10,:10,0]}\n")


    message = "Hello World".encode('utf8')
    print(f"Message to hide: {message}\n")

    hc = RC3(9, 4)
    print(f"Shared matrix H:\n{hc.H}\n")

    stego = cover.copy()
    stego[:,:,0] = hc.embed(cover[:,:,0], message)
    print(f"Stego:\n{stego[:10,:10,0]}\n")

    imageio.imsave("stego.png", stego)

    stego = imageio.imread("stego.png")
    extracted_message = hc.extract(stego[:,:,0])
    print("Extracted message:", extracted_message.decode())
 
    #print( (stego.astype('int16')-cover.astype('int16'))[:10,:10,0] )





