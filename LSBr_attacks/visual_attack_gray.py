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
		if pixels[x, y]%2==1:
			pixels[x, y] = 0
		else:
			pixels[x, y] = 255

i.show()
i.save(sys.argv[2])

