#!/usr/bin/env python3

import numpy
import sys


def prepare_M(n_bits):
    M=[]
    l=len(bin(2**n_bits-1)[2:])
    for i in range(1, 2**n_bits):
        string=bin(i)[2:].zfill(l)
        V=[]
        for c in string:
            V.append(int(c))
        M.append(V)
    M=numpy.array(M).T

    return M

def ME_hide_block(M, c, m):
    r=m-M.dot(c)
    r=r%2

    idx=0
    found=False
    for i in M.T:
        if numpy.array_equal(i, r):
            found=True
            break
        idx+=1

    # the block does not need to be modified
    if not found:
        return c

    s=numpy.array(c)
    if s[idx]==0: s[idx]=1
    else: s[idx]=0

    return s

def ME_unhide_block(M, s):
    m=M.dot(s)
    m=m%2
    return m








if __name__ == "__main__":

    # TODO: use key to mix matrix

    print("------------------")
    n_bits=3
    M=prepare_M(n_bits)
    m=numpy.array([1, 1, 0])
    print("m real=", m)
    c=numpy.array([0, 1, 1, 0, 1, 0, 0])
    s=ME_hide_block(M, c, m)
    print("cover:", c)
    print("stego:", s)
    m_recovered=ME_unhide_block(M, s)
    print("m_recovered=", m_recovered)



    print("------------------")
    n_bits=4
    M=prepare_M(n_bits)
    m=numpy.array([1, 1, 0, 0])
    print("m real=", m)
    c=numpy.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    s=ME_hide_block(M, c, m)
    print("cover:", c)
    print("stego:", s)
    m_recovered=ME_unhide_block(M, s)
    print("m_recovered=", m_recovered)


    print("------------------")
    n_bits=5
    M=prepare_M(n_bits)
    m=numpy.array([1, 1, 0, 0, 1])
    print("m real=", m)
    c=numpy.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0])
    s=ME_hide_block(M, c, m)
    print("cover:", c)
    print("stego:", s)
    m_recovered=ME_unhide_block(M, s)
    print("m_recovered=", m_recovered)


    print("------------------")
    n_bits=6
    M=prepare_M(n_bits)
    m=numpy.array([1, 1, 0, 0, 1, 1])
    print("m real=", m)
    c=numpy.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0])
    s=ME_hide_block(M, c, m)
    print("cover:", c)
    print("stego:", s)
    m_recovered=ME_unhide_block(M, s)
    print("m_recovered=", m_recovered)






