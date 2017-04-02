#!/usr/bin/python

import sys
from PIL import Image

if len(sys.argv) < 2:
	print "%s <img>\n" % sys.argv[0]
	sys.exit(1)

i = Image.open(sys.argv[1]) 
pixels = i.load()
width, height = i.size

# Histogram
histogram = [0]*255
for y in range(height):
	for x in range(width):
		cur_pixel = pixels[x, y]
		histogram[cur_pixel]+=1

# Substract pairs
total=0
for y in xrange(1, len(histogram), 2):
	dif=abs(histogram[y-1]-histogram[y])
	total+=dif

print total

