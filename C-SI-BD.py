#!/usr/bin/env python
# encoding: utf-8
"""
Emulate epidemic process using community-based SI-BD model.
"""
from __future__ import print_function

import random
import math
import sys
import operator
import yaml
from collections import defaultdict


cfg = yaml.load(open(sys.argv[1], 'r').read())
fn1 = sys.argv[2] # file with communitites
fn2 = sys.argv[3] # file with edges
rate_in = sys.argv[4]
rate_out = sys.argv[5]
MAXITER = int(sys.argv[7])
Tmax = float(sys.argv[6])
maxtags = cfg['maxtags']
dbgprint = cfg['dbgprint']
mincluster = cfg['mincluster']
minsize = cfg['minsize']


# import modules & graph

known_nodes = set()

G_out_edges = defaultdict(set)
for l in open(fn2):
    if " " in l.replace("\t"," ") and "#" not in l:
        z = l.replace("\t"," ").strip().split(" ")
        if z[0] != z[1]:
            known_nodes.add(int(z[0]))
            known_nodes.add(int(z[1]))
            G_out_edges[int(z[0])].add(int(z[1]))
            G_out_edges[int(z[1])].add(int(z[0]))
       

M = defaultdict(set)
for l in open(fn1):
    node = int(l.strip().split("\t")[0])
    if node in known_nodes:
        item = list(map(int,l.strip().split("\t")))
        M[item[1]].add(item[0])

M = [M[l] for l in M if len(M[l])>mincluster]
MM = defaultdict(int)
for i,m in enumerate(M):
    for j in m:
        MM[j] = i+1

total_seen_nodes = set()
total_seen_edges = set()
total_nodes_counter = 0
lambda_infect_in = float(rate_in)
lambda_infect_out = float(rate_out)
heal_time = Tmax
_iter = 0
while True:

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
        if current_time>Tmax: break
        trace.append("%d$%d$%f" %(from_node, current_node, current_time))
        new_nodes_counter += 1
        new_seen_nodes.add(current_node)
        if from_node != -1: 
            new_seen_edges.add((min(from_node,current_node),max(from_node,current_node)))

        recovered.add(current_node)
        infected = infected[1:]
        m_idx = MM[current_node] - 1
        seen_modules.add(m_idx)

        if dbgprint: print(current_node,'by',from_node,'ill since',current_time,'till',heal_time)
        new_nodes = set(MM.keys()) - recovered
        tagsdone = 0
        for node in new_nodes:
            effective_lambda = lambda_infect_in if m_idx+1 == MM[node] else lambda_infect_out
            infection_time = current_time + random.expovariate(effective_lambda)
            if infection_time > heal_time: continue
            if tagsdone > maxtags and maxtags>0: continue
            tagsdone += 1
            found = False
            for i,queued in enumerate(infected):
                if queued[1]==node:
                    if queued[0]>infection_time: 
                        infected[i][0] = infection_time
                        infected[i][2] = current_node
                        if dbgprint: print("# node ",node," was re-infected from",queued,"to",infected[i])
                        
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
        out = ["#",str(lambda_infect_in)+"/"+str(lambda_infect_out),_iter,len(trace)-2,len(seen_modules),total_nodes_counter,len(total_seen_nodes),len(total_seen_edges)]
        out.extend(trace)
        print("\t".join(map(str,out)))
        _iter += 1
        if _iter >= MAXITER:
            break
