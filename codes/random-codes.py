#!/usr/bin/env python3

import numpy as np
import sys

# Embed n-k=4 bits in n pixels. 
# Payload 0.36
n = 11
k = 7

"""
# Embed n-k=4 bits in n pixels. 
# Payload 0.40
n = 10
k = 6
H = np.array([
    # Identity     # Random
    [1, 0, 0, 0,   1, 1, 0, 0, 1, 1],
    [0, 1, 0, 0,   0, 1, 1, 0, 0, 1],
    [0, 0, 1, 0,   0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1,   1, 0, 0, 1, 0, 0],
])
x = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1])
m = np.array([1, 0, 1, 1])
"""


# Embed n-k=4 bits in n pixels. 
# Payload 0.44
n = 9
k = 5
H = np.array([
    # Identity     # Random
    [1, 0, 0, 0,   1, 1, 0, 0, 1],
    [0, 1, 0, 0,   0, 1, 1, 0, 0],
    [0, 0, 1, 0,   0, 0, 1, 1, 1],
    [0, 0, 0, 1,   1, 0, 0, 1, 0],
])

x = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])
m = np.array([0, 0, 1, 1])


"""
# Embed n-k=7 bits in n pixels. 
# Payload 0.43
n = 16
k = 9
H = np.array([
    # Identity              # Random
    # (n-k) * (n-k)          (n-k) * k
    [1, 0, 0, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0,   0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0,   0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0,   0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0,   0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0,   0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1,   1, 1, 1, 1, 1, 1, 0, 0, 0],
])

x = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0 ,0 ,0, 0, 1, 1, 1])
m = np.array([0, 0, 0, 1, 0, 0, 0])
"""







print("m to hide:", m)
print("default m:", H.dot(x)%2)


# Lookup table
D = {}
for i in range(2**n):
    new_y = np.array([int(b) for b in list(bin(i)[2:].zfill(n))])
    new_m = H.dot(new_y)%2
    if new_m.tobytes() in D:
        D[new_m.tobytes()].append(new_y)
    else:
        D[new_m.tobytes()] = [new_y]
        

closest_idx = np.argmin(np.sum(np.array(D[m.tobytes()])!=x, axis=1))
best_y = D[m.tobytes()][closest_idx]


print("x:", x)
print("y:", best_y)
print("cost:", np.sum(x!=best_y))
print("m:", H.dot(best_y)%2)




