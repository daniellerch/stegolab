#!/usr/bin/python
import sys
import random
from PIL import Image

if len(sys.argv) < 3:
	print "%s <stego img> <message>\n" % sys.argv[0]
	sys.exit(1)

img = Image.open(sys.argv[1]) 
pixels = img.load()
width, height = img.size

f = open(sys.argv[2], 'w')

idx=0
bitidx=0
bitval=0
for j in range(height):
	for i in range(width):
		if bitidx==8:
			f.write(chr(bitval))
			bitidx=0
			bitval=0
		bitval |= (pixels[i, j]%2)<<bitidx
		bitidx+=1

f.close()

