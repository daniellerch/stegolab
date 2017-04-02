#!/usr/bin/python

import sys
import random
from PIL import Image

def to_bits(filename):
	array=[]
	f=open(filename, 'r')
	bytes = (ord(b) for b in f.read())
	for b in bytes:
		for i in xrange(8):
			array.append((b >> i) & 1)
	return array

if len(sys.argv) < 4:
	print "%s <cover img> <stego img> <message>\n" % sys.argv[0]
	sys.exit(1)

img = Image.open(sys.argv[1]) 
pixels = img.load()
width, height = img.size
bits=to_bits(sys.argv[3])

sign=[1, -1]
idx=0
for j in range(height):
	for i in range(width):
		k=sign[random.randint(0, 1)]
		if pixels[i, j]==0: k=1
		if pixels[i, j]==255: k=-1
		if pixels[i, j]%2!=bits[idx]:
			pixels[i, j]+=k
		idx+=1
img.save(sys.argv[2])


