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

def get_clusters(fn):
    groundtruth_partition = dict()
    i = 0
    for line in open(fn):
        node, cluster = map(int, line.rstrip().split("\t"))
        groundtruth_partition[node] = cluster
    return groundtruth_partition

# get ground truth graph and partition
groundtruth_partition = get_clusters("graph/graph.clusters")

algorithm = sys.argv[1] # cd.random, cd.dani
filename = algorithm

dir = os.path.dirname("results/")
if not os.path.exists(dir):
    os.makedirs(dir)
    
res_file = open("results/"+filename, "w")
res_time = open("results/"+filename+"_time", "w")
metrics = ['sub_jaccard','sub_nmi','sub_nmi_arithm','sub_fnmi','sub_fnmi_arithm','sub_F-measure','sub_pearson','jaccard_diff','nmi_fixed','nmi_fixed_arithm','fnmi_fixed','fnmi_fixed_arithm','F-measure_diff','pearson_v2','pearson_v3']

for size in [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 30000]:
    print(size, end=' ', flush=True)
    time0 = tm.time()
    res_file.write(str(size)+"\t")
    results = dict()
    flag = False

    ep_fn = "epidemics/"+str(size)
    known_nodes = set()
    
    dst = open("multitree.ep", "w")
    for l in open(ep_fn):
        zz = l.strip().split('\t')
    nodes = set()
    for l in open(ep_fn):
        z = l.strip().split('\t')
        epidemia = []
        for w in z[9:]:
            root,node,time = w.split("$")
            nodes.add(int(node))
            known_nodes.add(int(node))
    # for n in sorted(nodes):
    #     dst.write(str(n)+","+str(n)+"\n")
    for n in range(max(known_nodes)+1):
        dst.write(str(n)+","+str(n)+"\n")
    dst.write("\n")
    for l in open(ep_fn):
        z = l.strip().split('\t')
        epidemia = []
        for w in z[9:]:
            root,node,time = w.split("$")
            epidemia.append(node+","+str(time).replace(",","."))
            # epidemia.append(time)
        dst.write(";".join(epidemia)+"\n")
    dst.close()

    if algorithm == "cd.dani":
        try:
            # run multitree
            command = "../CommDiff-Package/DiffusionCommunity/main -i:multitree.ep -Y:input.DANI >> DANI.stdout 2>> DANI.stderr"
            subprocess.check_call(command, shell=True)

            # run multitree -ep '+str(lmd)+'
            command = 'java -classpath "../CommDiff-Package/src:" myDANIBased.DANIProgram -i multitree.ep -p input.DANI -o output.DANI.comunities -r output.DANI.result  -si 10 -ss 10  >> DANIbased.stdout 2>> DANIbased.stderr'
            subprocess.check_call(command, shell=True)
        except:
            print(command)
            flag = True
        name_tmp = "output.DANI.comunities"

    if algorithm == "cd.random":
         try:
            command = 'java -classpath "../CommDiff-Package/src:" myRandBased.RandProgram -i multitree.ep -o output.RANDOM.comunities -r output.RANDOM.result  -si 10 -ss 10  >> RANDOMbased.stdout 2>> RANDOMbased.stderr'
            subprocess.check_call(command, shell=True)
         except:
            print(command)
            flag = True
            name_tmp = "output.RANDOM.comunities"
            
    if flag == False:
        partition = dict()
        for comm,line in enumerate(open("output.RANDOM.comunities")):
            chunks = line.strip().split(' ')
            # if chunks and len(chunks)>1:
            for node in chunks:
                if int(node) in known_nodes:
                    partition[int(node)] = comm

        assert(set(partition).issubset(groundtruth_partition))
        assert(known_nodes==set(partition))
    
        scores = partition_metrics.compare_partitions_metrics(partition,groundtruth_partition)

        for metric in metrics:
            results[metric] = scores[metric]
     
    if flag:
        print('iteration failed')
        res_file.write("\n")
        res_file.flush()
        res_time.write(str(size)+"\n")
        continue

    for metric in metrics:
        res_file.write(str(results[metric])+"\t")

    res_file.write("\n")
    res_file.flush()
    time1 = tm.time()
    res_time.write(str(size)+"\t")
    res_time.write(str(time1-time0)+"\n")
print()
