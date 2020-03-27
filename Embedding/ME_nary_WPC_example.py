#!/usr/bin/python3

import sys
import sympy
import scipy.linalg
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

    random.seed(1)

    n = 3
    for p in [3]:
        print("\n-- Experiment p =", p, ", n =", n, "--")
        m = np.array([random.randint(0, n-1) for i in range(p)])
        c = np.array([random.randint(0, n-1) for i in range(n**p - 1)])
        w = np.array([random.randint(0, len(c)-1) for i in range(3)]) # wet columns
        
        print("m real=", m)
        print("cover:", c)
        print("wet pixels:", w)
        M=prepare_M(p, n)
        print(M)

        # reverse array to remove correctly
        w=w[::-1] 
        # Matrix with wet columns removed
        H=np.array(M)
        H=np.delete(H, w, 1)


        # Solve Hx=m-Mc
        r=m-M.dot(c)
        r=r%n

        # Now we have too many solutions and solve() does no work in this case
        # so we remove unnecessary columns before to call solve().
        #H2=np.delete(H, np.s_[H.shape[0]:], 1)
        H2=np.delete(H, np.s_[H.shape[0]-1:-1], 1)
        a, b = sympy.symbols('a b')
        a = sympy.Matrix(H)
        b = sympy.Matrix(r)
        print(a, b)
        #sol = np.array(a.LUsolve(b))[:,0]%n
        #print("->", sol)

        print(H2)
        x = scipy.linalg.solve(H2, r)%n
        print(x)

        # We need one solution so we have to add zeros (any solution worth)
        for i in range(len(x), len(H[0])):
            x=np.append(x, 0)

        v=np.array(x, dtype=int)

        # reverse array to insert correctly
        w=w[::-1] 
        # insert wet values
        for i in w:
            v=np.insert(v, i, 0) 

        s = (c+v)%2 
        print("s =", s)
       


        s=ME_hide_block(M, c, m, n)
        m_recovered=ME_unhide_block(M, s, n)
        print("stego:", s)
        print("m_recovered=", m_recovered)

        
