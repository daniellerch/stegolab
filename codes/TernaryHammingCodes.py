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


# Matrix Embedding using ternary Hamming codes.




import sys
import random
import numpy as np


class HC3:

    def __init__(self, p, min_value=0, max_value=255):
        # {{{
        self.code = 3 # ternary
        self.msg_len = p
        self.block_len = (self.code**p-1) // (3-1)
        self.max_value=max_value
        self.min_value=min_value
        self.H = self._prepare_H()
        # }}}
 
    def _prepare_H(self):
        # {{{
        n = self.msg_len
        M=[]
        l = len(np.base_repr(3**n-1, base=3))
        for i in range(1, 3**n):
            string = np.base_repr(i, base=3).zfill(l)
            V = [ int(c) for c in string ]
            lindep = False
            for col in M:
                if (np.array_equal(1*np.array(col)%self.code, V) or
                    np.array_equal(2*np.array(col)%self.code, V) ):
                    lindep = True
                    break
            if lindep:
                continue
            M.append(V)
        M=np.array(M).T
        return M
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

        l = (len(m_list)//8 + 1)*8
        while len(m_list)<l:
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
                print("ERROR: wrong cover length:", c_chunk)
                sys.exit(0)

            column = (self.H.dot(c_chunk)-m_chunk)%self.code

            # Find position of column in H
            position = 0
            for v in self.H.T:
                if np.array_equal(v, column): 
                    add = -1
                    if s[j:j+self.block_len][position] - 1 < self.min_value:
                        add = 2
                    s[j:j+self.block_len][position] = \
                        (s[j:j+self.block_len][position] + add)%self.code
                    break
                elif np.array_equal((v*2)%self.code, column):
                    add = 1
                    if s[j:j+self.block_len][position] + 1 > self.max_value:
                        add = -2
                    s[j:j+self.block_len][position] = \
                        (s[j:j+self.block_len][position] + add)%self.code
                    break
                position += 1

            i += self.msg_len
            j += self.block_len

        return s
        # }}}

    def embed(self, cover, message):
        # {{{
        shape = cover.shape
        c = cover.flatten()
        m_message = self._bytes_to_ternary(message)

        # Insert message length
        mx = 2**32-1
        mx_msg_len = mx.to_bytes(4, 'big')
        m = self._bytes_to_ternary(mx_msg_len)
        mx_m_len = len(m)

        l = len(m_message)
        msg_len = l.to_bytes(4, 'big')


        m = self._bytes_to_ternary(msg_len)


        while len(m) < mx_m_len:
            m = np.insert(m, 0, 0)

        # Left padding
        while len(m) % self.msg_len != 0:
            m = np.insert(m, 0, 0)

        c_len =(len(m)//self.msg_len)*self.block_len
        s_header = self._embed(c[:c_len], m)


        # Insert the message
        message_bytes = self._ternary_to_bytes(m_message)

        # Right padding
        while len(m_message)%self.msg_len != 0:
            m_message = np.append(m_message, 0)


        s_content = self._embed(c[c_len:], m_message)

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
        shape = cover.shape
        s = stego.flatten()

        # Read header
        msg_len = (2**32-1).to_bytes(4, 'big')
        l = len(self._bytes_to_ternary(msg_len))
        while l%self.msg_len != 0:
            l += 1

        bytes_len =(l//self.msg_len)*self.block_len

        m = self._extract(s[:bytes_len])

        b = self._ternary_to_bytes(m)
        msg_len = int.from_bytes(b, 'big')

        # Read message
        m = self._extract(s[bytes_len:])
        while len(m)%8 != 0:
            m.pop()

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


    message = "Hello World asdas das".encode('utf8')
    print(f"Message to hide: {message}\n")

    hc = HC3(3)
    print(f"Shared matrix H:\n{hc.H}\n")

    stego = cover.copy()
    stego[:,:,0] = hc.embed(cover[:,:,0], message)
    print(f"Stego:\n{stego[:10,:10,0]}\n")

    imageio.imsave("stego.png", stego)

    stego = imageio.imread("stego.png")
    extracted_message = hc.extract(stego[:,:,0])
    print("Extracted message:", extracted_message.decode())
 





