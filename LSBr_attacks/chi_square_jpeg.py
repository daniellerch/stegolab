#!/usr/bin/python

import sys
import numpy
from PIL import Image
from scipy import stats


if len(sys.argv) < 2:
	print "%s <img>\n" % sys.argv[0]
	sys.exit(1)

f = open(sys.argv[1], "r" )
coef = []
for line in f:
    coef.append(int(line))

# loop over 1%, 2%, etc
last_pval=0
l=0
for sz in range(1,100):
	histogram = [0]*256

	# We use the percentage in sz
	l=int(len(coef)*(float(sz)/100))
	y=coef[1:l]
	for v in y:
		idx=v+128

		if idx>=256 or idx<=0:
			continue

		histogram[idx]=histogram[idx]+1

	# We get even and odd bins
	obs = numpy.array([])
	exp = numpy.array([])

	for y in range(1, len(histogram)/2):
		x=histogram[128-y]
		z=(histogram[128-y]+histogram[128+y])/2

		if x>0 and z>0:
			obs = numpy.append(obs, [x])
			exp = numpy.append(exp, [z])

	# Get chi square and p-value
	chi,pval = stats.chisquare(obs, exp)
	pval = 1-pval

	if pval<=0.01:
		if last_pval==0:
			print "Cover"
			exit()
		else:
			print last_pval
			break

	last_pval=pval;

print "Stego, length: %d" % sz



