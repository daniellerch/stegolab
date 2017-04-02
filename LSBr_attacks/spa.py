#!/usr/bin/python

import sys
from PIL import Image
from cmath import sqrt


if len(sys.argv) < 2:
	print "%s <img>\n" % sys.argv[0]
	sys.exit(1)


img = Image.open(sys.argv[1]) 
pixels = img.load()
width, height = img.size

x=0; y=0; k=0
for j in range(height):
	for i in range(width-1):
		r = pixels[i, j]
		s = pixels[i+1, j]
		if (s%2==0 and r<s) or (s%2==1 and r>s):
			x+=1
		if (s%2==0 and r>s) or (s%2==1 and r<s):
			y+=1
		if round(s/2)==round(r/2):
			k+=1

if k==0:
	print "SPA failed"
	exit

a=2*k
b=2*(2*x-width*(height-1))
c=y-x

bp=(-b+sqrt(b**2-4*a*c))/(2*a)
bm=(-b-sqrt(b**2-4*a*c))/(2*a)

beta=min(bp.real, bm.real)
if beta > 0.05:
	print "stego"
	print "Estimated embedding rate: %f" % beta
else:
	print "cover"


