#!/usr/bin/env python3
import sys
import subprocess
import networkx as nx

n = sys.argv[1]

#-maxk 20 -minc 30 -maxc 50

command = "./benchmark -N {} -t1 2.5 -t2 1.5 -mu 0.1 -k 5 -maxk 100 -minc 100 -maxc 600".format(n).split()
print(command)
subprocess.check_call(command)
        
graph = nx.read_edgelist("network.dat", nodetype=int)
with open("community.dat") as f:
    for line in f:
        if len(line.strip()) == 0:
            continue
        node, community = line.split()
        graph.node[int(node)]["community"] = int(community)
nx.write_graphml(graph, "lfr_graph_0.2_{}.graphml".format(n))
