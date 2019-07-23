#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import sys 
import random
from math import exp, log, sqrt
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import datetime
import cProfile
import community_ext as ce
import networkx as nx
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from community_ext.community_status import Status

import scipy.optimize
import numpy as np
import time as tm

__MIN = 0.0000001
__PASS_MAX = -1

start_ts = int(round(tm.time() * 1000))

fn = 'football.l0.1.epid1k.sample2.tsv'
if len(sys.argv)>1:
    fn = sys.argv[1]
# fn2 = fn.split('.')[0]+'.clusters'
# if len(sys.argv)>2:
#     fn2 = sys.argv[2]

# gt_clusters = dict()
# for l in open(fn2):
#     n,c = l.strip().split('\t')
#     gt_clusters[int(n)] = int(c)

seed = 0xfeeddead

# if len(sys.argv)>3:
#     seed = int(sys.argv[3])

np.random.seed(seed)
random.seed(seed)

method = 'ilfrs'
work_par_name = 'mu'
work_par = .5

debug = False
# debug = True

EPS = 1e-6
INF = 1e12
MAXPASS = -1

seen_nodes = set()
epidemics = []
_T_pauses_count = 0
_T_pauses_sum = 0.0
neighbours = defaultdict(set)

# load epidemics
for l in open(fn):
    z = l.strip().split('\t')
    epidemia = []
    for w in z[9:]:
        root,node,time = w.split("$")
        node = int(node)
        time = float(time)
        seen_nodes.add(node)
        epidemia.append((node,time))
    for n1,_ in epidemia:
        for n2,_ in epidemia:
            neighbours[n1].add(n2)
    epidemics.append(epidemia) # it's important, we have each epidemia already sorted by time in the generator script
    if len(epidemia)>1:
        #pauses = [epidemia[i][1]-epidemia[i-1][1] for i in range(1,len(epidemia))]
        pause = epidemia[len(epidemia)-1][1]-epidemia[len(epidemia)-2][1]
        _T_pauses_count += 1
        _T_pauses_sum += pause
        #_T_pauses_count += len(pauses)
        #_T_pauses_sum += sum(pauses)

average_delay = _T_pauses_sum/_T_pauses_count

def __randomly(seq, randomize=False):
    """ Convert sequence or iterable to an iterable in random order if
    randomize """
    if randomize:
        shuffled = list(seq)
        random.shuffle(shuffled)
        return iter(shuffled)
    return sorted(seq)

def trivial_clusters(nodes):
    return dict( [(n,c) for c,n in enumerate(nodes)] )

def trivial_tree(epidemics):
    edges = set()
    for epidemia in epidemics:
        if len(epidemia)>1:
            root = epidemia[0][0]
            for node,ts in epidemia[1:]:
                edges.add( (root,node) )
        else:
            edges.add( (epidemia[0][0],epidemia[0][0]) )

    dg = defaultdict(set)
    for from_node, to_node in edges:
        dg[from_node].add(to_node)
        dg[to_node].add(from_node)
    return dg

def trivial_chain(epidemics):
    edges = set()
    for epidemia in epidemics:
        if len(epidemia)>1:
            for i in range(1,len(epidemia)):
                edges.add( (epidemia[i-1][0],epidemia[i][0]) )
        else:
            edges.add( (epidemia[0][0],epidemia[0][0]) )
    dg = defaultdict(set)
    for from_node, to_node in edges:
        dg[from_node].add(to_node)
        dg[to_node].add(from_node)
    return dg

def dict2graph(dg):
    G = nx.Graph()
    for from_node, targets in dg.items():
        G.add_node(int(from_node))
        for to_node in targets:
            G.add_node(int(to_node))
            # if from_node not in G or to_node not in G[from_node]:
            if from_node != to_node:
                G.add_edge(int(from_node),int(to_node))
    return G

def ce__get_safe_par(model,pars=None):
    if not pars: 
        par = 1.-__MIN
    else:
        try:
            par = pars.get('mu',1.)
        except:
            par = 1.
        par = max(par,__MIN)
        par = min(par,1.-__MIN)
    return par

def ce__estimate_mu(graph,partition):
    Eout = 0
    Gsize = graph.size()
    for n1,n2 in graph.edges(): #links:
        if partition[n1] != partition[n2]: 
            Eout += 1
    return float(Eout)/Gsize

# def ce__get_DLD(status):
#     return sum(map(lambda x:x*log(x),filter(lambda x:x>0,status.rawnode2degree.values())))

def ce__get_es(status):
    E = float(status.total_weight)
    Ein = 0
    degrees_squared = 0.
    for community in set(status.node2com.values()):
        Ein += status.internals.get(community, 0.)
        tmp = status.degrees.get(community, 0.)
        degrees_squared += tmp*tmp #status.degrees.get(community, 0.)**2
    Eout = max(0.,E - Ein)
    return E,Ein,Eout,degrees_squared

def ce__modularity(status,model='ilfrs',pars = None):
    par = ce__get_safe_par(model,pars)
    E,Ein,Eout,_ = ce__get_es(status)
    result = 0.
    par = max(par,__MIN)
    par = min(par,1.-__MIN)
    result += Eout * log( par )
    result += Ein * log( 1 - par )
    result += - Eout * log( 2*E )
    result -= E
    # result_old1 = result
    # print('old1',result,'Eout',Eout,'Ein',Ein,'E',E,'par',par)
    result += sum(map(lambda x:x*log(x),filter(lambda x:x>0,status.rawnode2degree.values()))) # ce__get_DLD(status)
    # print('old2',result,result-result_old1)
    for community in set(status.node2com.values()):
        degree = status.degrees.get(community, 0.)
        if degree>0:
            result -= status.internals.get(community, 0.)*log(degree)
    # print('old3',result)
    return result

def ce__model_log_likelihood(graph,part_init,model,weight='weight',pars=None):
    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    return ce__modularity(status,model=model,pars={'mu':ce__estimate_mu(graph,part_init)})

def ce__transit(partition,rawnodepart):
    res = dict()
    for n,mn in rawnodepart.items():
        res[n] = partition[mn]
    return res

def ce__renumber(dictionary):
    count = 0
    ret = dictionary.copy()
    new_values = dict([])
    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value
    return ret

def ce_induced_graph(partition, graph, weight="weight"):
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())
    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})
    return ret

def ce_best_partition(graph, model=None, partition=None,
                   weight='weight', resolution=1., randomize=False, pars = None):
    part_init = partition
    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    ce__one_level(current_graph, status, weight, resolution, randomize, model=model, pars=pars) # TBD
    new_mod = ce__modularity(status,model=model,pars=pars)
    partition = ce__renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = ce_induced_graph(partition, current_graph, weight)

    status.init(current_graph, weight, raw_partition = ce__transit(partition,status.rawnode2node), raw_graph=graph)

    while True:
        ce__one_level(current_graph, status, weight, resolution, randomize, model=model, pars=pars) # TBD
        new_mod = ce__modularity(status,model=model,pars=pars)
        if new_mod - mod < __MIN:
            break
        partition = ce__renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = ce_induced_graph(partition, current_graph, weight)

        status.init(current_graph, weight, raw_partition = ce__transit(partition,status.rawnode2node), raw_graph=graph)
    dendrogram = status_list[:]
    partition = dendrogram[0].copy()
    for index in range(1, len(dendrogram)):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def ce__neighcom(node, graph, status, weight_key):
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight
    return weights

def ce__remove(node, com, weight, status):
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.degrees[-1] = status.degrees.get(-1, 0.)+status.gdegrees.get(node, 0.)
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1
    status.internals[-1] = status.loops.get(node, 0.)
    status.com2size[com] -= status.node2size[node]

def ce__insert(node, com, weight, status):
    status.node2com[node] = com
    status.degrees[-1] = status.degrees.get(-1, 0.)-status.gdegrees.get(node, 0.)
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))
    status.com2size[com] += status.node2size[node]

def ce__one_level(graph, status, weight_key, resolution, randomize, model='ppm', pars = None):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = ce__modularity(status,model=model,pars=pars)
    new_mod = cur_mod
    par = ce__get_safe_par(model,pars)
    __E = float(status.total_weight)
    __2E = 2.*__E
    mpar = 1.-par
    # __l2E = 0.
    __l2Epar = 0.
    __l2Epar2 = 0.
    if __E>0:
        # __l2E = log(__2E)
        if mpar>0.:
            __l2Epar = log(__2E*mpar/par)
    if mpar>0.:
        __l2Epar2 = (log(par/mpar)-log(__2E)) #__l2E)
    # __lpar = log(par)
    # __l2Epar3 = (__lpar-__l2E)
    # __par2E = par/__2E
    # P2 = len(status.rawnode2node)
    # P2 = P2*(P2-1)/2.
    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1
        for node in __randomly(graph.nodes(), randomize):
            com_node = status.node2com[node]
            neigh_communities = ce__neighcom(node, graph, status, weight_key)
            v_in_degree = neigh_communities.get(com_node,0)

            if model == 'ilfrs':
                com_degree = status.degrees.get(com_node, 0.)
                v_degree = status.gdegrees.get(node, 0.)
                v_loops  = graph.get_edge_data(node, node, default={weight_key: 0}).get(weight_key, 1)
                com_in_degree = status.internals.get(com_node, 0.)
                remove_cost = v_in_degree*__l2Epar2
                if com_degree>0.:
                    remove_cost += com_in_degree * log(com_degree)
                else:
                    remove_cost += com_in_degree / __MIN

                if com_degree > v_degree: remove_cost -= (com_in_degree - v_loops - v_in_degree) * log(com_degree - v_degree)
            ce__remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)

            best_com = com_node
            best_increase = 0.

            for com, dnc in __randomly(neigh_communities.items(),
                                       randomize):
                if model == 'ilfrs':
                    com_in_degree = status.internals.get(com, 0.)
                    com_degree = status.degrees.get(com, 0.)
                    add_cost = dnc*__l2Epar
                    add_cost += com_in_degree*log(com_degree)
                    add_cost -= (com_in_degree + v_loops + dnc) * log(com_degree + v_degree)

                incr = add_cost + remove_cost
                
                if incr > best_increase:
                    best_increase = incr
                    best_com = com

            ce__insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        if modified:
            new_mod = ce__modularity(status,model=model,pars=pars)
            if new_mod - cur_mod < __MIN:
                break


def score_graph_under_clusters(clusters, dg, pars=None):
    G = dict2graph(dg)
    # if pars is None:
    #     pars = {'mu': ce__estimate_mu(G,clusters)}
    #     print(pars)
    return ce__model_log_likelihood(G,clusters,method,weight='weight',pars=pars)


def optimize_clusters_under_graph(dg):
    G = dict2graph(dg)
    # work_par = 1.
    global work_par
    prev_par, it = work_par-1., 0
    prev_pars = set()
    while abs(work_par-prev_par)>1e-5: # stop if the size of improvement too small
        it += 1
        if it>100: break # stop after 100th iteration
        # update the parameter value
        prev_par = work_par
        if prev_par in prev_pars: break # stop if we are in the cycle
        prev_pars.add(prev_par)
        partition = ce_best_partition(G,model=method,pars={work_par_name:work_par})
        # calculate optimal parameter value for the current partition
        work_par = ce__estimate_mu(G,partition)
    loglike = ce__model_log_likelihood(G,partition,model=method,pars={work_par_name:work_par})
    return partition,loglike

def estimate_alpha(epidemics, dg):
    cnt = 0.
    cnt2 = 0.
    for epidemia in epidemics:
        Tmax = epidemia[-1][1] + average_delay/2
        infected_times = dict(epidemia)
        cnt += len(epidemia)-1
        for n1,edges in dg.items():
            t1 = infected_times.get(n1,Tmax)
            for n2 in edges:
                if n1 > n2: #  
                    t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                    # A1 -= alpha*abs(t1-t2)
                    cnt2 += abs(t1-t2)
    return cnt,cnt2

def precalc_estimate_alpha(epidemics, seen_nodes):
    edges_alpha_scores = defaultdict(float)
    for epidemia in epidemics:
        Tmax = epidemia[-1][1] + average_delay/2
        infected_times = dict(epidemia)
        for n1 in seen_nodes:
            t1 = infected_times.get(n1,Tmax)
            for n2 in seen_nodes:
                if n1 < n2: #  
                    t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                    if t1 != t2:
                        edges_alpha_scores[(n1,n2)] += abs(t1-t2)
    return edges_alpha_scores

# def precalc_A3(epidemics, graph):
#     A3_scores = defaultdict(float)
#     for epidemia in epidemics:
#         Tmax = 1 # epidemia[-1][1] + average_delay
#         infected_times = dict(epidemia)
#         if len(epidemia)>1:
#             prev_nodes = set()
#             prev_nodes.add(epidemia[0][0])
#             for n,t in epidemia[1:]: # for each node starting from the second
#                 for n2 in prev_nodes:
#                     linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n2,])))
#                     linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n2,])))
#                     if linked_prev_num_without:
#                         A3_scores[(min(n,n2),max(n,n2))] += log( linked_prev_num_with ) - log( linked_prev_num_without )
#                 prev_nodes.add(n)
#     return A3_scores

def precalc_graph_stats(epidemics, graph, seen_nodes):
    cnt = 0.
    cnt2 = 0.
    A3_scores = defaultdict(float)
    for epidemia in epidemics:
        Tmax = 1 # epidemia[-1][1] + average_delay
        infected_times = dict(epidemia)
        cnt += len(epidemia)-1
        for n1,edges in graph.items():
            t1 = infected_times.get(n1,Tmax)
            for n2 in edges:
                if n1 > n2: #  
                    t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                    # A1 -= alpha*abs(t1-t2)
                    cnt2 += abs(t1-t2)
        if len(epidemia)>1:
            prev_nodes = set()
            prev_nodes.add(epidemia[0][0])
            for n,t in epidemia[1:]: # for each node starting from the second
                for n2 in prev_nodes:
                    linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n2,])))
                    linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n2,])))
                    if linked_prev_num_without:
                        A3_scores[(min(n,n2),max(n,n2))] += log( linked_prev_num_with ) - log( linked_prev_num_without )
                    else:
                        A3_scores[(min(n,n2),max(n,n2))] += INF
                prev_nodes.add(n)
    return cnt, cnt2, A3_scores

# def precalc_A3(epidemics, graph):
#     for epidemia in epidemics:
#         Tmax = 1 # epidemia[-1][1] + average_delay
#         infected_times = dict(epidemia)
#         if len(epidemia)>1:
#             prev_nodes = set()
#             prev_nodes.add(epidemia[0][0])
#             for n,t in epidemia[1:]: # for each node starting from the second
#                 for n2 in prev_nodes:
#                     linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n2,])))
#                     linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n2,])))
#                     if linked_prev_num_without:
#                         A3_scores[(min(n,n2),max(n,n2))] += log( linked_prev_num_with ) - log( linked_prev_num_without )
#                 prev_nodes.add(n)
#     return A3_scores

# def get_A3(epidemics,graph,n1,n2):
#     A3 = 0.
#     # A3without = 0.
#     for epidemia in epidemics:
#         Tmax = 1 # epidemia[-1][1] + average_delay
#         infected_times = dict(epidemia)
#         if n1 not in infected_times and n2 not in infected_times: continue
#         # if n2 not in infected_times: continue
#         if len(epidemia)>1:
#             prev_nodes = set()
#             prev_nodes.add(epidemia[0][0])
#             for n,t in epidemia[1:]: # for each node starting from the second
#                 # if n not in (n1,n2): continue
#                 # linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n1,n2])))
#                 # linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n1,n2])))
#                 if n == n1:
#                     linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n2,])))
#                     linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n2,])))
#                 elif n == n2:
#                     linked_prev_num_with = len(prev_nodes & (set(graph[n]) | set([n1,])))
#                     linked_prev_num_without = len(prev_nodes & (set(graph[n]) - set([n1,])))
#                 else:
#                     linked_prev_num_with = len(prev_nodes & set(graph[n]))
#                     linked_prev_num_without = len(prev_nodes & set(graph[n]))
#                 if not linked_prev_num_without:
#                     return 0.,0.
#                 A3 += log( linked_prev_num_with )
#                 A3 -= log( linked_prev_num_without )
#                 prev_nodes.add(n)
#     return A3

def score_epidemics_under_graph(epidemics,dg,alphas=None):
    if alphas is None:
        alpha1,alpha2 = estimate_alpha(epidemics, dg)
    else:
        alpha1,alpha2 = alphas
        # edges_alpha_scores[n1,n2] = sum_epidemics( abs(infected_times.get(n1,Tmax)-infected_times.get(n2,Tmax)) )
        # при добавлении ребра из n1 в n2
        # alpha2 += edges_alpha_scores[n1,n2]
        # при удалении ребра из n1 в n2
        # alpha2 -= edges_alpha_scores[n1,n2]

    # при добавлении ребра из n1 в n2
    # A1 = X*(alpha1/alpha2)
    # A1*(alpha2/alpha1) = X 
    # A1' = A1*(alpha2/alpha1) * (alpha1/(alpha2+edges_alpha_scores[n1,n2])) - edges_alpha_scores[n1,n2]*alpha1/(alpha2+edges_alpha_scores[n1,n2])
    # A1' = A1*(alpha2/(alpha2+edges_alpha_scores[n1,n2])) - edges_alpha_scores[n1,n2]*alpha1/(alpha2+edges_alpha_scores[n1,n2])
    # A1' = (A1*alpha2 - edges_alpha_scores[n1,n2]*alpha1)/(alpha2+edges_alpha_scores[n1,n2]))
    # при удалении ребра из n1 в n2
    # A1' = (A1*alpha2 + edges_alpha_scores[n1,n2]*alpha1)/(alpha2-edges_alpha_scores[n1,n2]))

    # при добавлении ребра из n1 в n2
    # A2' = (A2/log(alpha))*log(alpha')
    # A2' = (A2/log(alpha1/alpha2))*log(alpha1/(alpha2+edges_alpha_scores[n1,n2])
    # при удалении ребра из n1 в n2
    # A2' = (A2/log(alpha1/alpha2))*log(alpha1/(alpha2-edges_alpha_scores[n1,n2])

    # A3 -- не придумал дешёвого способа быстрого апдейта

    alpha = alpha1/alpha2
    A1 = 0.
    A2 = 0.
    A3 = 0.
    for epidemia in epidemics:
        Tmax = 1 # epidemia[-1][1] + average_delay
        A2 += (len(epidemia)-1) # *log(alpha)
        infected_times = dict(epidemia)
        for n1,edges in dg.items():
            t1 = infected_times.get(n1,Tmax)
            for n2 in edges:
                if n1 > n2: #  
                    t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                    A1 -= alpha*abs(t1-t2)
        if len(epidemia)>1:
            prev_nodes = set()
            prev_nodes.add(epidemia[0][0])
            for n,t in epidemia[1:]: # for each node starting from the second
                linked_prev_num = len(prev_nodes & set(dg[n])) 
                if not linked_prev_num:
                    return -INF,0,0
                A3 += log( linked_prev_num )
                prev_nodes.add(n)
    # print("?",alpha1,alpha2)
    # print("??",A2*log(alpha),A2,log(alpha))
    return A1,A2*log(alpha),A3 #,A2

def score_epidemics_graph_clusters(epidemics,dg,partition):
    A1,A2,A3 = score_epidemics_under_graph(epidemics,dg)
    return A1+A2+A3+score_graph_under_clusters(partition, dg, pars=None)
    # AA = score_epidemics_under_graph(epidemics,dg)
    # return AA+score_graph_under_clusters(partition, dg)

current_graph = trivial_chain(epidemics)
current_clusters = trivial_clusters(seen_nodes)

try:
    grand_base_score = score_epidemics_graph_clusters(epidemics,current_graph,current_clusters)
    if debug: print('initial score is',grand_base_score)

    # if debug:
    #     GG = dict2graph(current_graph)
    #     spring_pos = nx.spring_layout(GG)
    #     values = [current_clusters.get(node) for node in GG.nodes()]
    #     nx.draw_networkx(GG, pos = spring_pos, cmap = plt.get_cmap("jet"), node_color = values, node_size = 55, with_labels = False)
    #     plt.show()

    edges_alpha_scores = precalc_estimate_alpha(epidemics, seen_nodes)

    while not False: # optimizing all
        if debug: print('graph optimization')
        current_score = grand_base_score
        base_score = grand_base_score

        # precalc cluster neighbours sets before internal cycle
        clusters_neighbours = defaultdict(set)
        for node,cluster in current_clusters.items():
            clusters_neighbours[cluster].add(node)

        while not False: # optimizing graph
            if debug: print('.graph optimization iteration')
            if debug: print('..adding edges')
            
            added_edges = set()
            candidates_to_add = []
            _a1,_a2,A3_scores =precalc_graph_stats(epidemics, current_graph, seen_nodes)
            A1,A2,A3 = score_epidemics_under_graph(epidemics, current_graph,(_a1,_a2))
            # parts of mu
            baseStatus = Status()
            baseStatus.init(dict2graph(current_graph), 'weight', current_clusters)
            baseE, baseEin, baseEout, _ = ce__get_es(baseStatus)
            base_degrees = sum(map(lambda x:x*log(x),filter(lambda x:x>0,baseStatus.rawnode2degree.values())))
            base_internals = 0.
            for community in set(baseStatus.node2com.values()):
                degree = baseStatus.degrees.get(community, 0.)    # это как-то меняется
                if degree>0:
                    base_internals -= baseStatus.internals.get(community, 0.)*log(degree)  # это как-то меняется            
            # print('===',baseE, baseEin, baseEout)
            baseMu = float(baseEout)/baseE
            newMu_same = float(baseEout)/(1.+baseE)
            newMu_diff = float(1.+baseEout)/(1.+baseE)

            for n in __randomly(seen_nodes, randomize=True):
                # best_node, best_score_change = None, 0.
                for n2 in (clusters_neighbours[current_clusters[n]]|neighbours[n])-current_graph[n]-set([n]):
                    if int(n)>=int(n2): continue

                    current_graph[n].add(n2)
                    current_graph[n2].add(n)

                    newE = baseE + 1
                    newEin = baseEin
                    newEout = baseEout
                    if current_clusters[n] == current_clusters[n2]:
                        newMu = newMu_same
                        newEin += 1
                    else:
                        newMu = newMu_diff
                        newEout += 1

                    baseMu_patched = min(max(baseMu,__MIN),1.-__MIN)
                    newMu_patched  = min(max(newMu,__MIN),1.-__MIN)
                    mod_diff = 0.

                    mod_diff += newEout*log(newMu_patched)
                    mod_diff += newEin*log(1 - newMu_patched)
                    mod_diff += -newEout*log(2*newE)
                    mod_diff += -newE
                    mod_diff += base_degrees
                    n1_degree = baseStatus.rawnode2degree.get(n,0)
                    if n1_degree>0: mod_diff -= n1_degree*log(n1_degree)
                    n1_degree += 1
                    mod_diff += n1_degree*log(n1_degree)
                    n2_degree = baseStatus.rawnode2degree.get(n2,0)
                    if n2_degree>0: mod_diff -= n2_degree*log(n2_degree)
                    n2_degree += 1
                    mod_diff += n2_degree*log(n2_degree)
                    mod_diff += base_internals
                    if current_clusters[n] == current_clusters[n2]:
                        c_degree = baseStatus.degrees.get(current_clusters[n], 0.)
                        c_internals = baseStatus.internals.get(current_clusters[n], 0.)
                        if c_degree:
                            mod_diff += c_internals*log(c_degree)
                        c_degree += 2
                        c_internals += 1
                        mod_diff -= c_internals*log(c_degree)
                    else:
                        c1_degree = baseStatus.degrees.get(current_clusters[n], 0.)
                        c1_internals = baseStatus.internals.get(current_clusters[n], 0.)
                        if c1_degree:
                            mod_diff += c1_internals*log(c1_degree)
                        c1_degree += 1
                        mod_diff -= c1_internals*log(c1_degree)
                        c2_degree = baseStatus.degrees.get(current_clusters[n2], 0.)
                        c2_internals = baseStatus.internals.get(current_clusters[n2], 0.)
                        if c2_degree:
                            mod_diff += c2_internals*log(c2_degree)
                        c2_degree += 1
                        mod_diff -= c2_internals*log(c2_degree)

                    _a2m = _a2 + edges_alpha_scores[(n,n2)]
                    A1m = (A1*_a2 + edges_alpha_scores[(n,n2)]*_a1)/(_a2-edges_alpha_scores[(n,n2)])
                    A2m = (A2/log(_a1/_a2))*log(_a1/(_a2+edges_alpha_scores[(n,n2)]))

                    score_change = A1m+A2m+A3+A3_scores[(n,n2)]+mod_diff-current_score

                    if score_change>0:
                        candidates_to_add.append( ((min(n,n2),max(n,n2)),score_change) )
                    current_graph[n].remove(n2)
                    current_graph[n2].remove(n)

            for i,((n,n2),score_change) in enumerate(sorted(candidates_to_add,key=lambda x:-x[1])):
                if debug: print('-',i,n2,score_change)
                current_graph[n].add(n2)
                current_graph[n2].add(n)
                if i:
                    actual_new_score = score_epidemics_graph_clusters(epidemics,current_graph,current_clusters)
                    # actual_new_score = score_epidemics_under_graph(epidemics, current_graph)+score_graph_under_clusters(current_clusters, current_graph) #-current_score
                    if debug: print('planned score diff is', score_change,'actual score diff is',actual_new_score-current_score)
                    if actual_new_score<current_score: 
                        if debug: print('skipped')
                        current_graph[n].remove(n2)
                        current_graph[n2].remove(n)
                        continue
                added_edges.add( (min(n,n2),max(n,n2)) )
                if i:
                    current_score = actual_new_score
                else:
                    current_score += score_change
                if debug: print('added, new score is ',current_score)
            if len(candidates_to_add):
                if debug: print('node',n,'done','current_score',current_score,'vs',score_epidemics_graph_clusters(epidemics,current_graph,current_clusters))
            if debug: print('..done adding edges')
            if debug: print('..removing edges')

            _a1,_a2,A3_scores =precalc_graph_stats(epidemics, current_graph, seen_nodes)
            A1,A2,A3 = score_epidemics_under_graph(epidemics, current_graph,(_a1,_a2))

            # parts of mu
            baseStatus = Status()
            baseStatus.init(dict2graph(current_graph), 'weight', current_clusters)
            baseE, baseEin, baseEout, _ = ce__get_es(baseStatus)
            base_degrees = sum(map(lambda x:x*log(x),filter(lambda x:x>0,baseStatus.rawnode2degree.values())))
            base_internals = 0.
            for community in set(baseStatus.node2com.values()):
                degree = baseStatus.degrees.get(community, 0.)    # это как-то меняется
                if degree>0:
                    base_internals -= baseStatus.internals.get(community, 0.)*log(degree)  # это как-то меняется            
            # print('===',baseE, baseEin, baseEout)
            baseMu = float(baseEout)/baseE
            newMu_same = float(baseEout)/(baseE-1.)
            newMu_diff = float(baseEout-1.)/(baseE-1.)
            candidates_to_remove = []

            for n in __randomly(seen_nodes, randomize=True):
                if not current_graph[n]: continue
                # best_node, best_score_change = None, 0.

                for n2 in (current_graph[n]-set([n])):
                    if int(n)>=int(n2): continue
                    if (min(n,n2),max(n,n2)) in added_edges: continue

                    current_graph[n].remove(n2)
                    current_graph[n2].remove(n)

                    newE = baseE - 1
                    newEin = baseEin
                    newEout = baseEout
                    if current_clusters[n] == current_clusters[n2]:
                        newMu = newMu_same
                        newEin -= 1
                    else:
                        newMu = newMu_diff
                        newEout -= 1

                    baseMu_patched = min(max(baseMu,__MIN),1.-__MIN)
                    newMu_patched  = min(max(newMu,__MIN),1.-__MIN)
                    mod_diff = 0.

                    mod_diff += newEout*log(newMu_patched)
                    mod_diff += newEin*log(1 - newMu_patched)
                    mod_diff += -newEout*log(2*newE)
                    mod_diff += -newE
                    mod_diff_opt1 = mod_diff

                    mod_diff += base_degrees
                    n1_degree = baseStatus.rawnode2degree.get(n,0)
                    mod_diff -= n1_degree*log(n1_degree)
                    n1_degree -= 1
                    if n1_degree>0: mod_diff += n1_degree*log(n1_degree)
                    n2_degree = baseStatus.rawnode2degree.get(n2,0)
                    mod_diff -= n2_degree*log(n2_degree)
                    n2_degree -= 1
                    if n2_degree>0: mod_diff += n2_degree*log(n2_degree)

                    mod_diff += base_internals
                    new_internals = base_internals
                    if current_clusters[n] == current_clusters[n2]:
                        c_degree = baseStatus.degrees.get(current_clusters[n], 0.)
                        c_internals = baseStatus.internals.get(current_clusters[n], 0.)
                        mod_diff += c_internals*log(c_degree)
                        new_internals += c_internals*log(c_degree)
                        c_degree -= 2
                        c_internals -= 1
                        if c_degree>0.:
                            mod_diff -= c_internals*log(c_degree)
                            new_internals -= c_internals*log(c_degree)
                    else:
                        c1_degree = baseStatus.degrees.get(current_clusters[n], 0.)
                        c1_internals = baseStatus.internals.get(current_clusters[n], 0.)
                        mod_diff += c1_internals*log(c1_degree)
                        new_internals += c1_internals*log(c1_degree)
                        c1_degree -= 1
                        if c1_degree:
                            mod_diff -= c1_internals*log(c1_degree)
                            new_internals -= c1_internals*log(c1_degree)
                        c2_degree = baseStatus.degrees.get(current_clusters[n2], 0.)
                        c2_internals = baseStatus.internals.get(current_clusters[n2], 0.)
                        mod_diff += c2_internals*log(c2_degree)
                        new_internals += c2_internals*log(c2_degree)
                        c2_degree -= 1
                        if c2_degree:
                            mod_diff -= c2_internals*log(c2_degree)
                            new_internals -= c2_internals*log(c2_degree)

                    _a2m = _a2 - edges_alpha_scores[(n,n2)]
                    A1m = (A1*_a2 - edges_alpha_scores[(n,n2)]*_a1)/(_a2+edges_alpha_scores[(n,n2)])
                    A2m = (A2/log(_a1/_a2))*log(_a1/(_a2-edges_alpha_scores[(n,n2)]))

                    score_change = A1m+A2m+A3-A3_scores[(n,n2)]+mod_diff-current_score

                    if score_change>0:
                        candidates_to_remove.append( ((min(n,n2),max(n,n2)),score_change) )
                    # if score_change>best_score_change:
                    #     best_node, best_score_change = n2, score_change
                    current_graph[n].add(n2)
                    current_graph[n2].add(n)

            while len(candidates_to_remove):
                if debug: print('Going to try to remove',len(candidates_to_remove),'candidate edges')
                _a1p,_a2p,A3_scores =precalc_graph_stats(epidemics, current_graph, seen_nodes)
                A1p,A2p,A3p = score_epidemics_under_graph(epidemics, current_graph,(_a1p,_a2p))
                removed_nodes = set()
                lost_candidates_to_remove = []
                for i,((n,n2),score_change) in enumerate(sorted(candidates_to_remove,key=lambda x:-x[1])):
                    if debug: print('-',i,'nodes',n,n2,'clusters',current_clusters[n],current_clusters[n2])
                    # _a1,_a2,_ =precalc_graph_stats(epidemics, current_graph, seen_nodes)
                    # A1,A2,A3 = score_epidemics_under_graph(epidemics, current_graph,(_a1,_a2))
                    # print('before removal scored as',score_epidemics_under_graph(epidemics, current_graph))
                    # print('As before removal:', _a1, _a2, A1, A2, A3, '=',sum((A1, A2, A3)))
                    # prA = sum((A1, A2, A3))
                    if n in removed_nodes or n2 in removed_nodes:
                        if debug: print('skipped because already touched')
                        lost_candidates_to_remove.append( ((n,n2),score_change) )
                        continue
                    current_graph[n].remove(n2)
                    current_graph[n2].remove(n)
                    _a2m = _a2p - edges_alpha_scores[(n,n2)]
                    A1m = (A1p*_a2p - edges_alpha_scores[(n,n2)]*_a1p)/(_a2p+edges_alpha_scores[(n,n2)])
                    A2m = (A2p/log(_a1p/_a2p))*log(_a1p/(_a2p-edges_alpha_scores[(n,n2)]))
                    A3m = A3p-A3_scores[(n,n2)]
                    _a1m = _a1p
                    # _a1,_a2,_ =precalc_graph_stats(epidemics, current_graph, seen_nodes)
                    # A1,A2,A3 = score_epidemics_under_graph(epidemics, current_graph,(_a1,_a2))
                    # print('removal scored as',score_epidemics_under_graph(epidemics, current_graph))
                    # print('As after removal:', _a1, _a2, A1, A2, A3, '=',sum((A1, A2, A3)))
                    # print('As opt    before:', _a1p, _a2p, A1p, A2p, A3p,'=',sum((A1p, A2p, A3p)))
                    # print('As opt     after:', _a1m, _a2m, A1m, A2m, A3m-A3_scores[(n,n2)],'=',sum((A1m, A2m, A3m)))
                    # if abs( (sum((A1m, A2m, A3m))-sum((A1p, A2p, A3p)))-(sum((A1, A2, A3))-prA) ) > __MIN and A3_scores[(n,n2)]<INF:
                    #     print('!!! As  opt diff:',sum((A1m, A2m, A3m))-sum((A1p, A2p, A3p)))
                    #     print('!!! As real diff:',sum((A1, A2, A3))-prA)
                    #     print('As after removal:', _a1, _a2, A1, A2, A3, '=',sum((A1, A2, A3)))
                    #     print('As opt    before:', _a1p, _a2p, A1p, A2p, A3p,'=',sum((A1p, A2p, A3p)))
                    #     print('As opt     after:', _a1m, _a2m, A1m, A2m, A3m-A3_scores[(n,n2)],'=',sum((A1m, A2m, A3m)))
                    #     print("???",A3m,A3_scores[(n,n2)])
                    # if i:
                    # actual_new_score = score_epidemics_graph_clusters(epidemics,current_graph,current_clusters)
                    actual_new_score = sum((A1m, A2m, A3m))+score_graph_under_clusters(current_clusters, current_graph)                
                        # actual_new_score = score_epidemics_under_graph(epidemics, current_graph)+score_graph_under_clusters(current_clusters, current_graph) #-current_score
                    if debug: print('planned score diff is', score_change,'actual score diff is',actual_new_score-current_score)
                    if actual_new_score<current_score: 
                        if debug: print('skipped')
                        current_graph[n].add(n2)
                        current_graph[n2].add(n)
                        continue
                    removed_nodes.add(n)
                    removed_nodes.add(n2)
                    _a1p = _a1m
                    _a2p = _a2m
                    A1p = A1m
                    A2p = A2m
                    A3p = A3m
                    current_score = actual_new_score
                    # if i:
                    #     current_score = actual_new_score
                    # else:
                    #     current_score += score_change
                    if debug: print('removed, new score is ',current_score)
                if len(lost_candidates_to_remove):
                    if debug: print('still need to try to remove',len(lost_candidates_to_remove),' edges')
                candidates_to_remove = lost_candidates_to_remove

            # if len(candidates_to_remove):
            #     if debug: print('node',n,'done',current_score,'vs',score_epidemics_graph_clusters(epidemics,current_graph,current_clusters))
            if debug: print('..done removing edges')
            if debug: print('.done with iteration',current_score)
            if current_score - base_score < EPS:
                break
            base_score = current_score

        if debug: print('done with graph optimization',current_score)
        graph_score = score_graph_under_clusters(current_clusters, current_graph)
        current_clusters, new_graph_score = optimize_clusters_under_graph(current_graph)
        current_score = score_epidemics_graph_clusters(epidemics,current_graph,current_clusters)
        if debug: print('graph score',graph_score,'->',new_graph_score)
        if debug: print('new clusterization done, score',current_score,', number of clusters',len(set(current_clusters.values())))
        # if debug: print(ce.compare_partitions(gt_clusters,current_clusters,safe=False))

        if debug: print('clusters cnt',Counter(current_clusters.values()),'clusters',current_clusters)
            # GG = dict2graph(current_graph)
            # values = [current_clusters.get(node) for node in GG.nodes()]
            # nx.draw_networkx(GG, pos = spring_pos, cmap = plt.get_cmap("jet"), node_color = values, node_size = 55, with_labels = False)
            # plt.show()

        if current_score - grand_base_score < EPS:
            break
        grand_base_score = current_score
except:
    # print("!!!",seen_nodes)
    current_clusters = dict()
    for comm,node in enumerate(sorted(seen_nodes)):
        current_clusters[node] = comm
    # exit()
# max_cluster = max(current_clusters.values())
# for n in gt_clusters.keys():
#     if n not in current_clusters.keys():
#         max_cluster += 1
#         current_clusters[n] = max_cluster
# results = ce.compare_partitions(gt_clusters,current_clusters)

# if sys.argv[4] == 'debag': 
# if debug: print('final clusters',current_clusters)
if debug: print(results)
total_time = int(round(tm.time() * 1000))-start_ts
if debug: print(total_time,'total milliseconds')
# print(current_clusters)
# print("\t".join(map(str,[sys.argv[0],fn,seed,total_time,results['nmi'],results['nmi_arithm'],results['rand'],results['jaccard']])))
fh = open('a13_results.tsv','w')
for node, cluster in sorted(current_clusters.items()):
    print('%d\t%d' % (node,cluster),file=fh)
fh.close()
