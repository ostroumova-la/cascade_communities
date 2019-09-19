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
algorithm = sys.argv[2] # cwnl.cr, cwnl.cic
iters_beg = int(sys.argv[3])
iters_end = int(sys.argv[4])
type = sys.argv[5] # auto, oracle
filename = epidemics + "_" + algorithm + "_" + type

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
        flag = False
        ep_fn = epidemics + "-" + str(j) + "/" + str(size)
        known_nodes = set()
        # print(algorithm)
        # exit()
        if algorithm == "cwnl.cr":
            # create epidemics in multitree format 
            max_node = 0
            for l in open("graph/graph.edges"):
                zz = l.strip().split('\t')
                max_node = max(max_node,int(zz[0]))
                max_node = max(max_node,int(zz[1]))

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

# -maxIt 100
            if type == "oracle":
                command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" diffusionBased.CommunityRate_Inference -a action.ep -c ../conf.inf -o output.cwnl.cr -k '+str(len(set(groundtruth_partition.values())))+'  -g graph/graph.clusters -l graph/graph.edges >> CR.stdout 2>> CR.stderr'
            else:
                command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" diffusionBased.CommunityRate_Inference_Annihilation -a action.ep -c ../conf.inf -o output.cwnl.cr -k '+str(len(known_nodes))+' >> CR.stdout 2>> CR.stderr'
                #print(len(set(groundtruth_partition.values())),"communities")
            try:
                subprocess.check_call(command, shell=True)
            except:
                flag = True
                #print(command)
                #exit()
                break

            partition = dict()
            comm2comm = dict()

            for line_num,line in enumerate(open("output.cwnl.crCommunities")):
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

        if algorithm == "cwnl.cic":
            # create epidemics in multitree format 
            max_node = 0
            for l in open("graph/graph.edges"):
                zz = l.strip().split('\t')
                max_node = max(max_node,int(zz[0]))
                max_node = max(max_node,int(zz[1]))

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


            if type == "oracle":
                command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" cicm.CommunityIC_Inference -a action.ep -c ../conf.inf -o output.cwnl.cic -k '+str(len(set(groundtruth_partition.values())))+' -g graph/graph.clusters -l graph/graph.edges >> CIC.stdout 2>> CIC.stderr'
            else:
                command = 'java -classpath "../CommunityWithoutNetworkLight/src:../CommunityWithoutNetworkLight/lib/utils.jar:../CommunityWithoutNetworkLight/lib/fastutil-6.4.2.jar" cicm.CommunityIC_Inference_Annihilation -a action.ep -c ../conf.inf -o output.cwnl.cic -k '+str(len(known_nodes))+' >> CIC.stdout 2>> CIC.stderr'
            try:
                subprocess.check_call(command, shell=True)
            except:
                flag = True
                #print(command)
                #exit()
                break

            partition = dict()
            comm2comm = dict()

            for line_num,line in enumerate(open("output.cwnl.cicCommunities")):
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
            results[metric].append(scores[metric]) 

    if flag == True:
        print('iteration failed')
        res_file.write("\n")
        res_file.flush()
        res_time.write(str(int(pow(2,size)))+"\n")
        continue
            
    #print(results)
    for metric in metrics:
        res_file.write(str(np.mean(results[metric]))+"\t")  
    res_file.write("\n")
    res_file.flush()
    time1 = tm.time()
    res_time.write(str(pow(2,size))+"\t")
    res_time.write(str(time1-time0)+"\n")

print()