#!/usr/bin/python

import sys
from PIL import Image


if len(sys.argv) < 2:
	print "%s <img>\n" % sys.argv[0]
	sys.exit(1)

i = Image.open(sys.argv[1]) 
pixels = i.load()
width, height = i.size

# Different valus for LSB=1 and LSB=0
for y in range(height):
	for x in range(width):
		r=g=b=0
		if pixels[x, y][0]%2==1: r=0
		else: r=255
		if pixels[x, y][1]%2==1: g=0
		else: g=255
		if pixels[x, y][2]%2==1: b=0
		else: b=255
		pixels[x, y] = (r, g, b)

i.show()
i.save(sys.argv[2])

