#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import partition_metrics
import networkx as nx
import community_ext
import sys
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
algorithm = sys.argv[2] # cd.random, cd.dani
filename = epidemics + "_" + algorithm
iters_beg = int(sys.argv[3])
iters_end = int(sys.argv[4])

res_file = open("results/"+filename, "w")
res_time = open("results/"+filename+"_time", "w")
metrics = ['sub_jaccard','sub_nmi','sub_nmi_arithm','sub_fnmi','sub_fnmi_arithm','sub_F-measure','sub_pearson','jaccard_diff','nmi_fixed','nmi_fixed_arithm','fnmi_fixed','fnmi_fixed_arithm','F-measure_diff','pearson_v2','pearson_v3']

for size in range(iters_beg,iters_end+1):
    print(size, end=' ', flush=True)
    time0 = tm.time()
    res_file.write(str((pow(2,size)))+"\t")
    results = defaultdict(lambda: [])
    
    for j in range(1,6):
        flag = False
        ep_fn = epidemics + "-" + str(j) + "/" + str(size)
        known_nodes = set()
        # print(algorithm)
        # exit()
        if algorithm == "cd.dani":
            # create epidemics in multitree format 
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

            try:
                # run multitree
                command = "../CommDiff-Package/DiffusionCommunity/main -i:multitree.ep -Y:input.DANI >> DANI.stdout 2>> DANI.stderr"
                subprocess.check_call(command, shell=True)

                # run multitree -ep '+str(lmd)+'
                command = 'java -classpath "../CommDiff-Package/src:" myDANIBased.DANIProgram -i multitree.ep -p input.DANI -o output.DANI.comunities -r output.DANI.result  -si 10 -ss 10  >> DANIbased.stdout 2>> DANIbased.stderr'
                subprocess.check_call(command, shell=True)
            except:
                print('ONE ITERATION FAILED')
                flag = True
                break
                
            partition = dict()
            for comm,line in enumerate(open("output.DANI.comunities")):
                chunks = line.strip().split(' ')
                # if chunks and len(chunks)>1:
                for node in chunks:
                    if int(node) in known_nodes:
                        partition[int(node)] = comm

        if algorithm == "cd.random":
            # create epidemics in multitree format 
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

            try:
                # run multitree -ep '+str(lmd)+'
                command = 'java -classpath "../CommDiff-Package/src:" myRandBased.RandProgram -i multitree.ep -o output.RANDOM.comunities -r output.RANDOM.result  -si 10 -ss 10  >> RANDOMbased.stdout 2>> RANDOMbased.stderr'
                subprocess.check_call(command, shell=True)
            except:
                print('ONE ITERATION FAILED')
                flag = True
                break
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
            results[metric].append(scores[metric])      
        
    if flag == True:
        res_file.write("\n")
        res_file.flush()
        res_time.write(str(int(pow(2,size)))+"\n")
        continue
    
    for metric in metrics:
        res_file.write(str(np.mean(results[metric]))+"\t")  
    res_file.write("\n")
    res_file.flush()
    time1 = tm.time()
    res_time.write(str(pow(2,size))+"\t")
    res_time.write(str(time1-time0)+"\n")

print()
