# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

if len(sys.argv)<3: 
	print('Usage: /me <input file> <minimum weight>')
	exit()

mw = float(sys.argv[2])

for line in open(sys.argv[1]):
	n1,n2,w = line.strip().split('\t')
	if float(w)>=mw:
		print("%s\t%s" % (n1,n2))