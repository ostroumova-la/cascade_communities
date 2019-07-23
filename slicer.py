#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import sys
import os

input_fn = "results"
ep_name = sys.argv[1]
range_from = int(sys.argv[2])
range_to = int(sys.argv[3])

# compute graph_edges
name = "graph/graph.edges"
edge_set = set()
for line in open(name, "r"):
    v1, v2 = line.split()
    if v1 != v2:
        edge_set.add(min(v1,v2)+"-"+max(v1,v2))
print(len(edge_set), "edges")
graph_edges = len(edge_set)

# generate range
use_range = [pow(2,i)*graph_edges for i in range(range_from,range_to+1)]

# open all necessary files

directory = os.path.dirname(ep_name)
if not os.path.exists(directory):
    os.makedirs(directory)

out_fhs = {}
for idx,slice in enumerate(use_range):
	out_fhs[slice] = open(ep_name+str(list(range(range_from,range_to+1))[idx]),'w')

# read epidemic file and write to slices
seen_edges = 0
for line in open(input_fn):
	chunks = line.strip().split("\t")
	for out_fh in out_fhs.values():
		out_fh.write(line)
	seen_edges += int(chunks[3])
	for limit in list(out_fhs.keys()):
		if limit <= seen_edges:
			out_fhs[limit].close()
			del(out_fhs[limit])

if len(out_fhs) > 0:
    print("Error, not enough data for", len(out_fhs),"slices")