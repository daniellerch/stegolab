#!/usr/bin/python

import sys
import numpy
from PIL import Image
from scipy import stats

if len(sys.argv) < 2:
	print "%s <img>\n" % sys.argv[0]
	sys.exit(1)

i = Image.open(sys.argv[1]) 
pixels = i.load()
width, height = i.size
chunk=1024
last_pval=0

# Read the image in chunks
for sz in xrange(chunk, height*width, chunk):

	# Obtain the histogram
	histogram = [0]*255
	for y in range(height):
		if y*height>sz:
			break
		for x in range(width):
			if type(pixels[x,y]) is tuple:
				for k in range(len(pixels[x, y])):
					cur_pixel = pixels[x, y][k]
					histogram[cur_pixel]+=1
			else:
				cur_pixel = pixels[x, y]
				histogram[cur_pixel]+=1

	# Odd and even bins
	obs = numpy.array([])
	exp = numpy.array([])
	X=0
	for y in xrange(1, len(histogram), 2):
		x=histogram[y-1]
		z=(histogram[y-1]+histogram[y])/2
		if x>0 and z>0:
			obs = numpy.append(obs, [x])
			exp = numpy.append(exp, [z])

	# Get chi square and p-value
	chi,pval = stats.chisquare(obs, exp)
	
	if pval<=0.01:
		if last_pval==0:
			print "Cover"
			exit()
		else:
			print last_pval
			break

	last_pval=pval;

print "Stego, length: %d" % sz

