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

def cos_sim(a, b, norm=True):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    if not norm: return dot_product
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_clusters(fn):
    groundtruth_partition = dict()
    i = 0
    for line in open(fn):
        node, cluster = map(int, line.rstrip().split("\t"))
        groundtruth_partition[node] = cluster
    return groundtruth_partition

# get ground truth graph and partition
groundtruth_partition = get_clusters("graph/graph.clusters")

algorithm = sys.argv[1] # oracle, multitree, path, wpath, clique, read
filename = algorithm

if algorithm == "clique":
    lmd0 = float(sys.argv[2])
    lmd = lmd0
    filename += "_"+str(lmd)
    
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
    if algorithm == "read":
        known_edges = set()
        for l in open(ep_fn):
            z = l.strip().split('\t')
            for t in z[9:]:
                zz = t.split("$")
                node_from = int(zz[0])
                node_to = int(zz[1])
                if node_from != -1:
                    known_edges.add( (str(min(node_from,node_to)),str(max(node_from,node_to))) )
                known_nodes.add(node_to)
    if algorithm == "multitree":
        # create epidemics in multitree format 
        dst = open("multitree.ep", "w")
        known_nodes = set()
        for l in open(ep_fn):
            z = l.strip().split('\t')
            epidemia = []
            for w in z[9:]:
                root,node,time = w.split("$")
                known_nodes.add(int(node))
        for n in sorted(known_nodes):
            dst.write(str(n)+","+str(n)+"\n")
        dst.write("\n")
        for l in open(ep_fn):
            z = l.strip().split('\t')
            epidemia = []
            for w in z[9:]:
                root,node,time = w.split("$")
                epidemia.append(node)
                epidemia.append(time)
            dst.write(",".join(epidemia)+"\n")
        dst.close()

        # run multitree
        seen_vertices = str(5*len(known_nodes))

        command = "../network-inference-multitree/network-inference-multitree -i:multitree.ep -e:"+seen_vertices+" -o:mtree > tmp"
        try:
            subprocess.check_call(command, shell=True)
        except:
            flag = True
            break
            
        # get obtained graph
        fn = "mtree-edge.info"
        known_edges = set()
        for i,l in enumerate(open(fn)):
            if not i: continue
            z = l.strip().split('/')
            node_from = int(z[0])
            node_to = int(z[1])
            known_edges.add( (str(min(node_from,node_to)),str(max(node_from,node_to))) )
    
    if algorithm == "clique":
        EPS = 0.000001
        
        if lmd0 == -1:
            sum = 0.
            num = 0
            for l in open(ep_fn):
                prev_time = 0.
                z = l.strip().split('\t')
                for t in z[10:]:
                    zz = t.split("$")
                    cur_time = float(zz[2])
                    num+=1
                    sum+=(cur_time-prev_time)
                    prev_time = cur_time
            lmd = num/sum
            
        if lmd0 == -2:
            sum = 0.
            num = 0
            for l in open(ep_fn):
                z = l.strip().split('\t')
                for i1 in range(10,len(z)):
                    zz = z[i1].split("$")
                    cur_time = float(zz[2])
                    for i2 in range(9,i1):
                        zz = z[i2].split("$")
                        prev_time = float(zz[2])
                        num+=1
                        sum+=(cur_time-prev_time)
            lmd = num/sum
        
        projected_cascade_sequence_lmd = defaultdict(float)
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0: continue
            seq = []
            for t in z[9:]:
                zz = t.split("$")
                time = float(zz[2])
                node_to = zz[1]
                seq.append([time,node_to])
                known_nodes.add(int(node_to))
            seq = sorted(seq,key=lambda x:x[0])

            norm_count = 0.
            t2_old = 0.
            for i2 in range(1,len(seq)):
                t2 = seq[i2][0]
                norm_count = (norm_count + 1.) * exp( -lmd * (t2-t2_old))
                t2_old = t2

                for i1 in range(0,i2):
                    t1 = seq[i1][0]
                    key = min(seq[i1][1],seq[i2][1])+"-"+max(seq[i1][1],seq[i2][1])
                    projected_cascade_sequence_lmd[key] += exp( -lmd * (t2-t1) ) / norm_count
                    
        known_edges = {}
        for t in projected_cascade_sequence_lmd.items():
            if t[1]<EPS: continue
            nn1,nn2 = t[0].split("-")
            known_edges[(nn1,nn2)] = t[1]
       
    if algorithm == "oracle":               
        known_edges = set()
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0:
                print("error!")
                exit(0)

            for t in z[9:]:
                zz = t.split("$")
                node_from = int(zz[0])
                node_to = int(zz[1])
                if node_from != -1:
                    known_edges.add( (str(min(node_from,node_to)),str(max(node_from,node_to))) )
                known_nodes.add(node_to)
                
    if algorithm == "path":
        known_edges = set()
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0: continue

            prev = -1
            for t in z[9:]:
                zz = t.split("$")
                node_from = prev
                node_to = int(zz[1])
                if node_from != -1:
                    known_edges.add( (str(min(node_from,node_to)),str(max(node_from,node_to))) )
                prev = node_to
                known_nodes.add(node_to)
                
    if algorithm == "ramezani":
        EPS = 0.000001
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0: continue
            for t in z[9:]:
                zz = t.split("$")
                node_to = int(zz[1])
                known_nodes.add(node_to)

        nodes_signatures = defaultdict(list)
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0: continue
            ep_nodes = set()
            for t in z[9:]:
                zz = t.split("$")
                node_to = int(zz[1])
                ep_nodes.add(node_to)

            for n in known_nodes:
                if n in ep_nodes:
                    nodes_signatures[n].append( 1. )
                else:
                    nodes_signatures[n].append( 0. )

        known_edges = dict()
        for n1 in known_nodes:
            for n2 in known_nodes:
                if int(n2)>int(n1):
                    value = cos_sim(nodes_signatures[n1],nodes_signatures[n2])
                    if value > EPS:
                        known_edges[ (str(n1),str(n2)) ] = value
                        
                
    if algorithm == "wpath":
        known_edges = defaultdict(int)
        for l in open(ep_fn):
            z = l.strip().split('\t')
            if int(z[3]) == 0: continue

            prev = -1
            for t in z[9:]:
                zz = t.split("$")
                node_from = prev
                node_to = int(zz[1])
                if node_from != -1:
                    known_edges[ (str(min(node_from,node_to)),str(max(node_from,node_to))) ] += 1
                prev = node_to
                known_nodes.add(node_to)
                
    # quit if read only
    if algorithm != "read":
       
        # create our graph
        G = nx.Graph()
        for v in known_nodes:
            G.add_node(v)
        for e in known_edges:
            if algorithm in ("oracle","multitree","path"):
                from_node, to_node = int(e[0]), int(e[1])
                G.add_edge(from_node,to_node)
                #G.add_edge(from_node,to_node,**{"weight": 1})                
            if algorithm in ("wpath","clique","ramezani"):
                from_node, to_node = int(e[0]), int(e[1])
                wght = float(known_edges[e])
                G.add_edge(from_node,to_node,**{"weight": wght})

        partition = community_ext.best_partition(G, model="dcppm", pars={'gamma':1.0}, randomize=True, weight="weight")
    
        assert(set(partition).issubset(groundtruth_partition))
        assert(known_nodes==set(partition))
    
        scores = partition_metrics.compare_partitions_metrics(partition,groundtruth_partition)

        for metric in metrics:
            results[metric] = scores[metric]
    
    if algorithm == "read":
       time1 = tm.time()
       res_time.write(str(size)+"\t")
       res_time.write(str(time1-time0)+"\n")
       res_file.write("\n")
       continue
    
    
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
