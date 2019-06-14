#!/usr/bin/env python
# encoding: utf-8
"""
Emulate epidemic process using SIR model.
"""
from __future__ import print_function

import random
import math
import sys
import operator
import yaml
from collections import defaultdict
import numpy as np

cfg = yaml.load(open(sys.argv[1], 'r'))
fn1 = sys.argv[2] # file with communitites
fn2 = sys.argv[3] # file with edges
rate_param = sys.argv[4] # parameter to generate infections

MAXITER = int(sys.argv[5])

maxtags = cfg['maxtags']
dbgprint = cfg['dbgprint']
mincluster = cfg['mincluster']
lambda_heal = cfg['lambda_heal']
minsize = cfg['minsize']

# import modules & graph

M = defaultdict(set)
for l in open(fn1):
    item = list(map(int,l.strip().split("\t")))
    M[item[1]].add(item[0])

M = [M[l] for l in M if len(M[l])>mincluster]
MM = defaultdict(int)
for i,m in enumerate(M):
    for j in m:
        MM[j] = i+1

G_out_edges = defaultdict(set)
for l in open(fn2):
    if " " in l.replace("\t"," ") and "#" not in l:
        z = l.replace("\t"," ").strip().split(" ")
        G_out_edges[int(z[0])].add(int(z[1]))
        G_out_edges[int(z[1])].add(int(z[0]))

total_seen_nodes = set()
total_seen_edges = set()
total_nodes_counter = 0


_iter = 0
while True:
    lambda_infect = np.random.pareto(float(rate_param))
    start_node = random.choice(list(G_out_edges.keys()))

    recovered = set()
    infected = list()
    infected.append( [0, start_node, -1] )
    seen_modules = set()
    trace = [start_node,]

    new_seen_nodes = set()
    new_seen_edges = set()
    new_nodes_counter = 0

    while len(infected)>0:
        current_time,current_node,from_node = infected[0]
        
        trace.append("%d$%d$%f" %(from_node, current_node, current_time))
        new_nodes_counter += 1
        new_seen_nodes.add(current_node)
        if from_node != -1: 
            new_seen_edges.add((min(from_node,current_node),max(from_node,current_node)))
        heal_time = current_time + random.expovariate(lambda_heal)
        recovered.add(current_node)
        infected = infected[1:]
        m_idx = MM[current_node] - 1
        seen_modules.add(m_idx)

        if dbgprint: print(current_node,'by',from_node,'ill since',current_time,'till',heal_time)
        new_nodes = set(G_out_edges[current_node]) - recovered
        tagsdone = 0
        for node in new_nodes:
        
            infection_time = current_time + random.expovariate(lambda_infect)
            if infection_time > heal_time: continue
            if tagsdone > maxtags and maxtags>0: continue
            tagsdone += 1
            found = False
            for i,queued in enumerate(infected):
                if queued[1]==node:
                    if queued[0]>infection_time: 
                        if dbgprint: print("# node ",node," was re-infected from",queued,"to",)
                        infected[i][0] = infection_time
                        infected[i][2] = current_node
                        if dbgprint: print(infected[i])
                    found = True
                    break

            if not found:
                infected.append( [infection_time, node, current_node] )
                if dbgprint: print ("# queued:",infected[-1])
        infected = sorted(infected, key=operator.itemgetter(0))

    if len(trace)-1 >= minsize:
        total_nodes_counter += new_nodes_counter
        total_seen_nodes |= new_seen_nodes
        total_seen_edges |= new_seen_edges
        out = ["#",lambda_infect,_iter,len(trace)-2,len(seen_modules),total_nodes_counter,len(total_seen_nodes),len(total_seen_edges)]
        out.extend(trace)
        print ("\t".join(map(str,out)))
        _iter += 1
        if _iter >= MAXITER:
            break
