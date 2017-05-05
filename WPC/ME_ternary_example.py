#!/usr/bin/python

import numpy
import sys


def ternary (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

def prepare_M(n):
    M=[]
    l=len(ternary(3**n-1))
    for i in range(1, 3**n):
        string=ternary(i).zfill(l)
        V=[]
        for c in string:
            V.append(int(c))
        M.append(V)
    M=numpy.array(M).T
    print M

    return M

def ME_hide_block(M, c, m):
    r=m-M.dot(c)
    r=r%3

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
    #if s[idx]==0: s[idx]=1
    #else: s[idx]=0
    s[idx]+=1

    return s

def ME_unhide_block(M, s):
    m=M.dot(s)
    m=m%3
    return m








if __name__ == "__main__":

    # TODO: use key to mix matrix

    print "------------------"
    n=3
    M=prepare_M(n)
    m=numpy.array([1, 2, 1])
    print "m real=", m
    c=numpy.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1 ,0])
    s=ME_hide_block(M, c, m)
    print "cover:", c
    print "stego:", s
    m_recovered=ME_unhide_block(M, s)
    print "m_recovered=", m_recovered



    print "------------------"
    n=4
    M=prepare_M(n)
    m=numpy.array([2, 1, 0, 2])
    print "m real=", m
    c=numpy.array([
        0, 1, 1, 0, 1, 0, 0, 1, 1, 
        0, 1, 1, 0, 1, 0, 0, 1, 1, 
        0, 0, 1, 0, 1, 0, 0, 1, 1, 
        0, 1, 1, 0, 1, 0, 1, 0, 1, 
        0, 1, 1, 0, 1, 0, 0, 1, 1, 
        0, 0, 1, 0, 1, 1, 0, 1, 1, 
        0, 1, 1, 0, 1, 0, 0, 0, 1, 
        0, 1, 1, 0, 1, 0, 0, 1, 0, 
        0, 1, 1, 0, 1, 0, 1, 1
        ])
    s=ME_hide_block(M, c, m)
    print "cover:", c
    print "stego:", s
    m_recovered=ME_unhide_block(M, s)
    print "m_recovered=", m_recovered









