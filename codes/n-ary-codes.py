#!/usr/bin/python3

import sys
import random
import numpy as np

def nary(p, n):
    if p == 0:
        return '0'
    nums = []
    while p:
        p, r = divmod(p, n)
        nums.append(str(r))
    return ''.join(reversed(nums))

def prepare_M(p, n):
    M=[]
    l=len(nary(n**p-1, n))
    for i in range(1, n**p):
        string=nary(i, n).zfill(l)
        V=[]
        for c in string:
            V.append(int(c))
        M.append(V)
    M=np.array(M).T

    return M

def ME_hide_block(M, c, m, n):
    r=m-M.dot(c)
    r=r%n

    idx=0
    found=False
    for i in M.T:
        if np.array_equal(i, r):
            found=True
            break
        idx+=1

    if not found:
        return c

    s=np.array(c)
    s[idx]+=1

    return s

def ME_unhide_block(M, s, n):
    m=M.dot(s)
    m=m%n
    return m








if __name__ == "__main__":

    random.seed(0)

    n = 3
    for p in [2, 3, 5]:
        print("\n-- Experiment p =", p, ", n =", n, "--")
        m = np.array([random.randint(0, n-1) for i in range(p)])
        c = np.array([random.randint(0, n-1) for i in range(n**p - 1)])
        print("m real=", m)
        print("cover:", c)
        M=prepare_M(p, n)
        print(M)
        s=ME_hide_block(M, c, m, n)
        m_recovered=ME_unhide_block(M, s, n)
        print("stego:", s)
        print("m_recovered=", m_recovered)

        
    n = 4
    for p in [2, 3, 5]:
        print("\n-- Experiment p =", p, ", n =", n, "--")
        m = np.array([random.randint(0, n-1) for i in range(p)])
        c = np.array([random.randint(0, n-1) for i in range(n**p - 1)])
        print("m real=", m)
        print("cover:", c)
        M=prepare_M(p, n)
        print(M)
        s=ME_hide_block(M, c, m, n)
        m_recovered=ME_unhide_block(M, s, n)
        print("stego:", s)
        print("m_recovered=", m_recovered)

    n = 5
    for p in [2, 3, 5]:
        print("\n-- Experiment p =", p, ", n =", n, "--")
        m = np.array([random.randint(0, n-1) for i in range(p)])
        c = np.array([random.randint(0, n-1) for i in range(n**p - 1)])
        print("m real=", m)
        print("cover:", c)
        M=prepare_M(p, n)
        print(M)
        s=ME_hide_block(M, c, m, n)
        m_recovered=ME_unhide_block(M, s, n)
        print("stego:", s)
        print("m_recovered=", m_recovered)








