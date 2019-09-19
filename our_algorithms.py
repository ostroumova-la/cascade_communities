#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import partition_metrics
import networkx as nx
import community_ext
import sys, os
from collections import defaultdict
import numpy as np
from math import pow, exp
import subprocess
import time as tm

def get_graph(fn):
    G = nx.Graph()
    for line in open(fn):
        from_node, to_node = map(int, line.rstrip().split("\t"))
        if from_node != to_node:
            if from_node not in G or to_node not in G[from_node]:
                G.add_edge(from_node,to_node)

    #get ground truth partition
    fn1 = fn.replace(".edges",".clusters")
    groundtruth_partition = dict()
    
    for line in open(fn1):
        node, cluster = map(int, line.rstrip().split("\t"))
        if node not in G.nodes(): continue
        groundtruth_partition[node] = cluster
    return G, groundtruth_partition

# get ground truth graph and partition
G0, groundtruth_partition = get_graph("graph/graph.edges")

epidemics = sys.argv[1] # type of epidemics
algorithm = sys.argv[2] # a13, a12
filename = epidemics + "_" + algorithm
iters_beg = int(sys.argv[3])
iters_end = int(sys.argv[4])
if algorithm == "a12":
    lmd0 = float(sys.argv[5])
    lmd = lmd0
    filename += "_"+str(lmd)
    
dir = os.path.dirname("results/")
if not os.path.exists(dir):
    os.makedirs(dir)
    
res_file = open("results/"+filename, "w")
res_time = open("results/"+filename+"_time", "w")
metrics = ['sub_jaccard','sub_nmi','sub_nmi_arithm','sub_fnmi','sub_fnmi_arithm','sub_F-measure','sub_pearson','jaccard_diff','nmi_fixed','nmi_fixed_arithm','fnmi_fixed','fnmi_fixed_arithm','F-measure_diff','pearson_v2','pearson_v3']

for size in range(iters_beg,iters_end+1):
    print(size, end=' ', flush=True)
    time0 = tm.time()
    res_file.write(str((pow(2,size)))+"\t")
    results = defaultdict(lambda: [])
    
    for j in range(1,6):
        ep_fn = epidemics + "-" + str(j) + "/" + str(size)
        known_nodes = set()
        if algorithm == "a13":
            # run multitree
            command = "python3 ../stub_alg1.3.py "+ep_fn
            subprocess.check_call(command, shell=True)

            partition = dict()
            for line in open("a13_results.tsv"):
                if line.strip():
                    node, comm = line.strip().split("\t")
                    partition[int(node)] = int(comm)
                    known_nodes.add(int(node))
  
        if algorithm == "a12":
            EPS = 0.000001
            
            if lmd0 == -1:
                sum = 0.
                num = 0
                # first variant
                for l in open(ep_fn):
                    prev_time = 0.
                    z = l.strip().split('\t')
                    for t in z[10:]:
                        zz = t.split("$")
                        cur_time = float(zz[2])
                        num+=1
                        sum+=(cur_time-prev_time)
                        prev_time = cur_time
                if sum:
                    lmd = num/sum
                else:
                    lmd = 0.
                #print(lmd)
                
            if lmd0 == -2:
                sum = 0.
                num = 0
                # first variant
                for l in open(ep_fn):
                    z = l.strip().split('\t')
                    for i1 in range(10,len(z)):
                        t = z[i1]
                        zz = t.split("$")
                        cur_time = float(zz[2])
                        for i2 in range(9,i1):
                            t = z[i2]
                            zz = t.split("$")
                            prev_time = float(zz[2])
                            num+=1
                            sum+=(cur_time-prev_time)
                if sum:
                    lmd = num/sum
                else:
                    lmd = 0.
                #print(lmd)
            
            projected_cascade_sequence_lmd = defaultdict(float)
            for l in open(ep_fn):
                z = l.strip().split('\t')
                if int(z[3]) == 1: continue
                seq = []
                for t in z[9:]:
                    zz = t.split("$")
                    seq.append([float(zz[2]),zz[1]])
                    known_nodes.add(int(zz[1]))
                seq = sorted(seq,key=lambda x:x[0])

                for i2 in range(1,len(seq)):
                    t2 = seq[i2][0]
                    to_update = []
                    norm_count = 0.

                    for i1 in range(0,i2):
                        t1 = seq[i1][0]
                        norm_count += exp( -lmd * (t2-t1) )
                        key = min(seq[i1][1],seq[i2][1])+"-"+max(seq[i1][1],seq[i2][1])
                        to_update.append( (key,exp( -lmd * (t2-t1) )) )
                    for key,count in to_update:
                        projected_cascade_sequence_lmd[key] += count / norm_count
                        
            known_edges = {}
            for t in projected_cascade_sequence_lmd.items():
                if t[1]<EPS: continue
                nn1,nn2 = t[0].split("-")
                known_edges[(nn1,nn2)] = t[1]
            # print(known_edges)
            # create our graph
            G = nx.Graph()
            for e in known_edges:
                from_node, to_node = int(e[0]), int(e[1])
                wght = float(known_edges[e])
                G.add_edge(from_node,to_node,**{"weight": wght})
            # print(G)
            initial_partition = community_ext.best_partition(G, model="dcppm", pars={'gamma':1.0}, randomize=True, weight="weight")
            fh = open('a12_input.tsv','w')
            for node, cluster in sorted(initial_partition.items()):
                print('%d\t%d' % (node,cluster),file=fh)
            fh.close()

            command = "python3 ../stub_alg1.2_1step.py "+ep_fn+" a12_input.tsv"
            subprocess.check_call(command, shell=True)

            partition = dict()
            for line in open("a12_results.tsv"):
                if line.strip():
                    node, comm = line.strip().split("\t")
                    partition[int(node)] = int(comm)
                    known_nodes.add(int(node))

        
        assert(set(partition).issubset(groundtruth_partition))
        assert(known_nodes==set(partition))

        scores = partition_metrics.compare_partitions_metrics(partition,groundtruth_partition)

        for metric in metrics:
            results[metric].append(scores[metric])   
        
    for metric in metrics:
        res_file.write(str(np.mean(results[metric]))+"\t")

    res_file.write("\n")
    res_file.flush()
    time1 = tm.time()
    res_time.write(str(pow(2,size))+"\t")
    res_time.write(str(time1-time0)+"\n")
    
print()

