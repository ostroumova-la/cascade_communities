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

algorithm = sys.argv[1] # cwnl.cr, cwnl.cic
type = sys.argv[2] # auto, oracle
filename = algorithm + "_" + type

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
    

    dst = open("action.ep", "w")
    dst.write("vertexId\ttraceId\ttimestamp\tInfluencer\n")
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
    for ep_n,l in enumerate(open(ep_fn)):
        z = l.strip().split('\t')
        epidemia = []
        for w in z[9:]:
            root,node,time = w.split("$")
            if root == '-1': 
                root = '-'
            epidemia.append(str(int(node))+"\t"+str(ep_n)+"\t"+str(int(1e3*float(time)))+"\t"+root)
        dst.write("\n".join(epidemia)+"\n")
    epidemia = []
    for n in groundtruth_partition.keys():
        if n not in known_nodes:
            ep_n += 1
            epidemia.append(str(n)+"\t"+str(ep_n)+"\t"+str(0)+"\t"+'-')
    dst.write("\n".join(epidemia)+"\n")
    dst.close()

    if algorithm == "cwnl.cr":
        if type == "oracle":
            command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" diffusionBased.CommunityRate_Inference -a action.ep -c ../conf.inf -o output.cwnl.cr -k '+str(len(set(groundtruth_partition.values())))+'  -g graph/graph.clusters -l graph/graph.edges >> CR.stdout 2>> CR.stderr'
        else:
            command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" diffusionBased.CommunityRate_Inference_Annihilation -a action.ep -c ../conf.inf -o output.cwnl.cr -k '+str(len(known_nodes))+' >> CR.stdout 2>> CR.stderr'
            #print(len(set(groundtruth_partition.values())),"communities")

    if algorithm == "cwnl.cic":
         if type == "oracle":
            command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" cicm.CommunityIC_Inference -a action.ep -c ../conf.inf -o output.cwnl.cic -k '+str(len(set(groundtruth_partition.values())))+' -g graph/graph.clusters -l graph/graph.edges >> CIC.stdout 2>> CIC.stderr'
         else:
            command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" cicm.CommunityIC_Inference_Annihilation -a action.ep -c ../conf.inf -o output.cwnl.cic -k '+str(len(known_nodes))+' >> CIC.stdout 2>> CIC.stderr'
    try:
        subprocess.check_call(command, shell=True)
    except:
        flag = True
        print(command)     
            
    if flag == False:
        partition = dict()
        comm2comm = dict()
        
        if algorithm == "cwnl.cr":
            name_tmp = "output.cwnl.crCommunities"
        if algorithm == "cwnl.cic":
            name_tmp = "output.cwnl.cicCommunities"
        
        for line_num,line in enumerate(open(name_tmp)):
            if line_num:
                node,comm = line.strip().split('\t')
                node = int(node)
                if node in known_nodes:
                    if comm not in comm2comm:
                        next_comm = 0
                        if len(comm2comm):
                            next_comm = max(comm2comm.values())+1
                        comm2comm[comm] = next_comm
                    comm_num = comm2comm[comm]
                    partition[node] = comm_num

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
