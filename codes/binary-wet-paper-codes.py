#!/usr/bin/python3

import numpy
import sys
import scipy.linalg 


if __name__ == "__main__":

    # Shared matrix
    M=numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
    , dtype=int)

    # Wet pixels: columns to remove
    # Only sender knows
    w=numpy.array([2, 3, 4, 6, 11, 12, 13, 14])

    # reverse array to remove correctly
    w=w[::-1] 

    # Matrix with wet columns removed
    H=numpy.array(M)
    H=numpy.delete(H, w, 1)

    # Message
    m=numpy.array([0, 1, 1, 0])
    m=numpy.array([0, 0, 1, 1])

    # Cover pixels
    c=numpy.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0])

    print("M=", M)
    print("H=", H)
    print("m=", m)
    print("w=", w)
    print("c=", c)

    # Solve Hx=m-Mc
    r=m-M.dot(c)

    # Now we have too many solutions and solve() does no work in this case
    # so we remove unnecessary columns before to call solve().
    H2=numpy.delete(H, numpy.s_[H.shape[0]:], 1)
    x = scipy.linalg.solve(H2, r)%2

    # We need one solution so we have to add zeros (any solution worth)
    for i in range(len(x), len(H[0])):
        x=numpy.append(x, 0)

    v=numpy.array(x, dtype=int)

    # reverse array to insert correctly
    w=w[::-1] 
    # insert wet values
    for i in w:
        v=numpy.insert(v, i, 0) 

    s = (c+v)%2 
    print("s=", s)
    

    # RECEIVER
    m=M.dot(s)
    m=m%2
    print("m recovered=", m)




