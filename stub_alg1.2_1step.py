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

#random.seed(0)
import scipy.optimize
import numpy as np

# fn = 'football.l0.1.epid1k.sample2.tsv'
# fn = 'polblogs.l0.02.epid_cover.sample2.tsv'
if len(sys.argv)>1:
    fn = sys.argv[1]
# fn2 = fn.split('.')[0]+'.clusters'
# if len(sys.argv)>2:
#     fn2 = sys.argv[2]
fn3 = sys.argv[2]

EPS = 1e-6
MAXPASS = 1000

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
        pauses = [epidemia[i][1]-epidemia[i-1][1] for i in range(1,len(epidemia))]
        _T_pauses_count += len(pauses)
        _T_pauses_sum += sum(pauses)
        
def trivial_clusters(nodes):
    return dict( [(n,c) for c,n in enumerate(nodes)] )

def _eta(data):
    """ Compute eta for NMI calculation
    """
    # if len(data) <= 1: return 0
    ldata = len(list(data))
    if ldata <= 1: return 0
    _exp = exp(1)
    counts = Counter()
    for d in data:
        counts[d] += 1
    # probs = [float(c) / len(data) for c in counts.values()]
    probs = [float(c) / ldata for c in counts.values()]
    probs = [p for p in probs if p > 0.]
    ent = 0
    for p in probs:
        if p > 0.:
            ent -= p * log(p, _exp)
    return ent

def _nmi(x, y):
    """ Calculate NMI without including extra libraries
    """
    # print(x,y)
    # print(list(x))
    sum_mi = 0.0
    x_value_list = list(set(x))
    y_value_list = list(set(y))
    lx = len(list(x))
    ly = len(list(y))
    Px = []
    for xval in x_value_list:
        # Px.append( len(filter(lambda q:q==xval,x))/float(lx) )
        Px.append( len(list(filter(lambda q:q==xval,x)))/float(lx) )
    Py = []
    for yval in y_value_list:
        # Py.append( len(filter(lambda q:q==yval,y))/float(ly) )
        Py.append( len(list(filter(lambda q:q==yval,y)))/float(ly) )

    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = []
        for j,yj in enumerate(y):
            if x[j] == x_value_list[i]:
                sy.append( yj )
        if len(sy)== 0:
            continue
        pxy = []
        for yval in y_value_list:
            # pxy.append( len(filter(lambda q:q==yval,sy))/float(ly) )
            pxy.append( len(list(filter(lambda q:q==yval,sy)))/float(ly) )

        t = []
        for j,q in enumerate(Py):
            if q>0:
                t.append( pxy[j]/Py[j] / Px[i])
            else:
                t.append( -1 )
            if t[-1]>0:
                sum_mi += (pxy[j]*log( t[-1]) )
    eta_xy = _eta(x)*_eta(y)
    if eta_xy == 0.: return 0
    return 2*sum_mi/(_eta(x)+_eta(y))

def compare_partitions(p1,p2):
    """Compute three metrics of two partitions similarity:
      * Rand index
      * Jaccard index
      * NMI 

    Parameters
    ----------
    p1 : dict
       a first partition
    p2 : dict
       a second partition

    Returns
    -------
    r : dict
       with keys 'rand', 'jaccard', 'nmi'

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> part1 = best_partition(G, model='ppm', pars={'gamma':0.5})
    >>> part2 = best_partition(G, model='dcppm', pars={'gamma':0.5})
    >>> compare_partitions(part1, part2)
    """
    p1_sets = defaultdict(set)
    p2_sets = defaultdict(set)
    [p1_sets[item[1]].add(item[0]) for item in p1.items()]
    [p2_sets[item[1]].add(item[0]) for item in p2.items()]
    p1_sets = p1_sets.values()
    p2_sets = p2_sets.values()
    cross_tab = [[0, 0], [0, 0]]
    for a1, s1 in enumerate(p1_sets):
        for a2, s2 in enumerate(p2_sets):
            common = len(s1 & s2)
            l1 = len(s1) - common
            l2 = len(s2) - common
            cross_tab[0][0] += common * (common - 1)
            cross_tab[1][0] += common * l2
            cross_tab[0][1] += l1 * common
            cross_tab[1][1] += l1 * l2
    [[a00, a01], [a10, a11]] = cross_tab
    # print(p1)
    # print(p2)
    p1_vec = list(map(lambda x:x[1],sorted(p1.items(),key=lambda x:x[0])))
    p2_vec = list(map(lambda x:x[1],sorted(p2.items(),key=lambda x:x[0])))
    # p1_vec = list(p1_vec)[:]
    # print(list(p1_vec)[:])
    return {
        'nmi': _nmi(p1_vec,p2_vec),
        'rand': float(a00 + a11) / (a00 + a01 + a10 + a11),
        'jaccard' : float(a00) / (a01 + a10 + a00)
    }

EPS = 0.000001

def __randomly(seq, randomize=False):
    """ Convert sequence or iterable to an iterable in random order if
    randomize """
    if randomize:
        shuffled = list(seq)
        random.shuffle(shuffled)
        return iter(shuffled)
    return sorted(seq)

def naive_loglike_with_gradients(clusters, epidemics, p_in, p_out):
    nodes = clusters.keys()
    active_clusters = Counter(clusters.values())
    A1 = 0.
    A2 = 0.
    dA1din = 0.
    dA2din = 0.
    dA1dout = 0.
    dA2dout = 0.

    for epidemia in epidemics:
        Tmax = epidemia[-1][1] + average_delay # current T as the end of times
        infected_times = dict(epidemia)
        if len(epidemia)>1:
            clusters_seen = defaultdict(int) # it's a counter of nodes seen so far in each cluster and in total 
            clusters_seen[clusters[epidemia[0][0]]] += 1 # count the root node cluster
            clusters_seen['any'] += 1 
            for i,(n1,t1) in enumerate(epidemia): # the outer cycle is only by infected nodes
                c1 = clusters.get(n1)
                # for n2 in nodes: # the inner cycle goes through all nodes
                for n2,c2 in clusters.items():
                    if n1 >= n2 and n2 in infected_times: continue # here we eliminated duplicated pairs such as (n1,n2) and (n2,n1)
                    t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                    if c1 == c2: 
                        A1 += -p_in*abs(t1-t2)
                        dA1din += -abs(t1-t2)
                    else:
                        A1 += -p_out*abs(t1-t2)
                        dA1dout += -abs(t1-t2)
                if i:
                    A2 += log(p_in * clusters_seen[c1] + p_out * ( clusters_seen['any'] - clusters_seen[c1] ) )
                    dA2din  += clusters_seen[c1] / float( p_in*clusters_seen[c1] + p_out*(clusters_seen['any']-clusters_seen[c1]) )
                    dA2dout += (clusters_seen['any']-clusters_seen[c1]) / float( p_in*clusters_seen[c1] + p_out*(clusters_seen['any']-clusters_seen[c1]) )
                    clusters_seen[c1] += 1
                    clusters_seen['any'] += 1
        else:
            n1 = epidemia[0][0]
            t1 = 0.
            t2 = average_delay
            c1 = clusters.get(n1)
            for c2,cnt in active_clusters.items():
                if c1 == c2: 
                    A1 += -p_in*average_delay*cnt
                    dA1din += -average_delay*cnt
                else:
                    A1 += -p_out*average_delay*cnt
                    dA1dout += -average_delay*cnt
            A1 -= -p_in*average_delay
            dA1din -= -average_delay
    return A1+A2,(dA1din+dA2din),(dA1dout+dA2dout)

def naive_loglike(clusters, epidemics, p_in, p_out):
    return naive_loglike_with_gradients(clusters, epidemics, p_in, p_out)[0]

def estimate_base_alpha(clusters, epidemics):
    nodes = clusters.keys()
    A1 = 0.
    A2 = 0.
    for epidemia in epidemics:
        Tmax = epidemia[-1][1] + average_delay # current T as the end of times
        infected_times = dict(epidemia)
        for n1,t1 in epidemia: # the outer cycle is only by infected nodes
            for n2 in nodes: # the inner cycle goes through all nodes
                if n1 >= n2 and n2 in infected_times: continue # here we eliminated duplicated pairs such as (n1,n2) and (n2,n1)
                t2 = infected_times.get(n2,Tmax) # here we applied Tmax for all uninfected nodes
                A1 += abs(t1-t2)
        A2 += len(epidemia)-1
    return A2/float(A1)


def greedy_optimize_clusters(clusters, epidemics, p_in, p_out, randomize = False):
    nodes = clusters.keys()
    # print('#',datetime.datetime.now(),'optimization started')
    sys.stdout.flush()
    # precalculate all what possible
    p_diff = p_in - p_out
    nodes_number = len(nodes)
    nontrivial_Tmaxes = []
    nontrivial_epidemics = []
    trivial_epidemics_by_node = defaultdict(float)
    n2n_A1part = defaultdict(float)
    for epidemia in epidemics:
        if len(epidemia)>1:
            infected_times = OrderedDict(epidemia)
            Tmax = epidemia[-1][1] + average_delay
            nontrivial_Tmaxes.append( Tmax )
            nontrivial_epidemics.append( infected_times )
            for n1,t1 in infected_times.items():
                for n2 in nodes: #range(i1+1,nodes_number):
                    if n1 >= n2 and n2 in infected_times: continue # here we eliminated duplicated pairs such as (n1,n2) and (n2,n1)
                    t2 = infected_times.get(n2, Tmax)
                    td = abs(t1-t2)
                    n2n_A1part[(n1,n2)] += td
                    n2n_A1part[(n2,n1)] += td
        else:
            trivial_epidemics_by_node[epidemia[0][0]] += 1.
    # for infected_times,Tmax in zip(nontrivial_epidemics,nontrivial_Tmaxes):
    # print('#',datetime.datetime.now(),'precalculation done')
    sys.stdout.flush()

    # init starting iteration conditions
    cur_score,_,_ = naive_loglike_with_gradients(clusters, epidemics, p_in, p_out)
    # print('#',datetime.datetime.now(),'basic loglike calculated')
    sys.stdout.flush()
    modified = True
    cur_iter = 0
    while modified and cur_iter != MAXPASS:
        cur_iter += 1
        new_score = cur_score
        for qqq,n1 in enumerate(nodes):
            c1 = clusters[n1]
            # print('node',n1,cur_iter,qqq)
            trivial_n1 = trivial_epidemics_by_node[n1]
            active_clusters = Counter(clusters.values())
            A1part = defaultdict(int)
            A2part = defaultdict(int)
            A3part = defaultdict(int)
            # calc diff A1
            A3part[c1] -= 2*trivial_n1
            for n2,c2 in clusters.items():
                # A1part[c2] += n2n_A1part[(n1,n2)]
                # A3part[c2] += trivial_epidemics_by_node[n2]+trivial_n1
                A1part[c2] += n2n_A1part.get( (n1,n2), 0. )
                A3part[c2] += trivial_epidemics_by_node.get( n2, 0. )+trivial_n1
            # calc diff A2 (it defined only for nontrivial epidemics)
            for infected_times in nontrivial_epidemics:
                if n1 in infected_times: # если вершина $i$ не заражена, то дифф равен нулю, если вершина $i$ заражена, то
                    t1 = infected_times.get(n1)
                    seen_before_n1 = defaultdict(int)
                    seen_after_n1  = defaultdict(int)
                    for n2, t2 in infected_times.items(): # note! it's ordered dict, and it's important
                        c2 = clusters[n2]
                        if t2<t1: # сюда входят вершины раньше $i$
                            seen_before_n1[c2] += 1
                            seen_before_n1['any'] += 1
                        else: # сюда входят вершины начиная с $i$
                            seen_after_n1[c2] += 1  
                            seen_after_n1['any'] += 1
                        if t2>t1: # изменение для всех вершин, которые заразились после $i$ 
                            patch = 1
                            if c2 != c1: patch = -1
                            A2part[c2] += -log(
                                    p_in*(seen_before_n1[c2]+seen_after_n1[c2]-1) + # -1, т.к. мы только что посчитали tk в seen_after_n1[c2]
                                    p_out*(seen_before_n1['any']+seen_after_n1['any']-seen_before_n1[c2]-seen_after_n1[c2])
                                ) + log(
                                    p_in*(seen_before_n1[c2]+seen_after_n1[c2]-1-patch) +
                                    p_out*(seen_before_n1['any']+seen_after_n1['any']-seen_before_n1[c2]-seen_after_n1[c2]+patch)
                                )
                    if seen_before_n1['any']>0:
                        for cluster in active_clusters:
                            patch = 1
                            if cluster != c1: patch = -1
                            A2part[cluster] += patch*(log(p_out*seen_before_n1['any']) - log(p_in*seen_before_n1[cluster] \
                                                      + p_out*(seen_before_n1['any']-seen_before_n1[cluster])))

            remove_cost = p_diff*(A1part[c1]+average_delay*A3part[c1]) + A2part[c1]
            best_cluster, best_cluster_score = None,0.
            for cluster in active_clusters:
                add_cost = -p_diff*(A1part[cluster]+average_delay*A3part[cluster]) + A2part[cluster]
                 # print('..',n1,'from',c1,'to',cluster,'=',remove_cost+add_cost)
                if remove_cost+add_cost>best_cluster_score:
                    best_cluster_score = remove_cost+add_cost
                    best_cluster = cluster
            if best_cluster is not None:
                 # print('best cluster for node',n1,'is',best_cluster,best_cluster_score,'current',c1)
                 clusters[n1] = best_cluster
                 new_score += best_cluster_score
                 modified = True
            # exit()
        if modified:
            # new_score = naive_loglike(clusters, epidemics, p_in, p_out)
            # print('#',datetime.datetime.now(),'iteration done, ll changed from',cur_score,'to',new_score,'total clusters left',len(set(clusters.values())))
            sys.stdout.flush()
            if new_score - cur_score < EPS:
                break
            cur_score = new_score
    return clusters,cur_score

def optimize_p(clusters, epidemics, p_in, p_out, par,dbg = False):
    clusters = clusters
    epidemics = epidemics
    dbg = dbg
    _cache = dict()
    if par not in ('p_in','p_out'):
        print('ERR: invalid par')
        exit()
    def closure(x):
        x = x[0] 
        if x.__class__.__name__ == 'ndarray': # DIRTY HACK FOR L-BFGS-B DIMS COMPATIBILITY
            x = x[0]
        if dbg:    print("...x:",x)
        if x in _cache: 
            res,der = _cache[x]
        else:
            if par == 'p_in':
                res,der,_ = naive_loglike_with_gradients(clusters,epidemics,x,p_out)
            elif par == 'p_out':
                res,_,der = naive_loglike_with_gradients(clusters,epidemics,p_in,x)
            _cache[x] = (res,der)
        if dbg:    print("...f,df:",res,der)
        return -res,np.array(-der)
    if par == 'p_in':
        opt = scipy.optimize.minimize(closure, [p_in], jac=True, method='L-BFGS-B', bounds=((EPS,1.-EPS),))
    elif par == 'p_out':
        opt = scipy.optimize.minimize(closure, [p_out], jac=True, method='L-BFGS-B', bounds=((EPS,1.-EPS),))
    return opt['x'][0], -opt['fun']

try:
    #print("try")
# if 1:
    average_delay = _T_pauses_sum/_T_pauses_count
    # print('seen nodes:', len(seen_nodes))
    # init clusters
    clusters = dict()
    for l in open(fn3):
        n,c = l.strip().split('\t')
        clusters[int(n)] = int(c)
    # get ground truth to compare with
    # gt_clusters = dict()
    # for l in open(fn2):
    #     n,c = l.strip().split('\t')
    #     gt_clusters[int(n)] = int(c)
    # print('#',datetime.datetime.now(),'graph loaded')
    sys.stdout.flush()
    # basic estimation of p_out
    p_base = estimate_base_alpha(clusters,epidemics)
    p_in, p_out = 10*p_base,p_base
    # print('#',datetime.datetime.now(),'p_base calculated')
    # print('p_base',p_base,'p_in',p_in,'p_out',p_out)
    sys.stdout.flush()

    last_p_opt_ll = naive_loglike(clusters,epidemics,p_in,p_out)
    while 1: # now optimize p_out and p_in until change will be less than EPS
        #print('#',datetime.datetime.now(),'optimizing alphas:')
        sys.stdout.flush()
        p_out, new_p_opt_ll = optimize_p(clusters, epidemics, p_in, p_out, 'p_out') #,dbg = True)
        p_in,  new_p_opt_ll = optimize_p(clusters, epidemics, p_in, p_out, 'p_in') #,dbg = True)
        # print('new p_in',p_in,'new p_out',p_out, '\ncurrent ll is', new_p_opt_ll)
        sys.stdout.flush()
        if new_p_opt_ll - last_p_opt_ll < EPS:
            last_p_opt_ll = new_p_opt_ll
            break
        last_p_opt_ll = new_p_opt_ll
    clusters, cl_opt_ll = greedy_optimize_clusters(clusters, epidemics, p_in, p_out)
except:
    # print("!!!",seen_nodes)
    clusters = dict()
    for comm,node in enumerate(sorted(seen_nodes)):
        clusters[node] = comm

# print('#',datetime.datetime.now(),'initial alpha optimization done:')
# print('\tp_in',p_in,'p_out',p_out)
# sys.stdout.flush()
# print('gt ll:', naive_loglike(gt_clusters,epidemics,p_in,p_out))
# sys.stdout.flush()
# print('ni ll:', naive_loglike(clusters,epidemics,p_in,p_out))
# sys.stdout.flush()
# # clusters = trivial_clusters(seen_nodes)
# current_ll = naive_loglike(clusters,epidemics,p_in,p_out)
# print('solo ll:', current_ll)
# print('--- opt started ---')
# sys.stdout.flush()

# while 1: # iterate until change will be less than EPS
#     # optimize clusters with given p_in, p_out
# print('#',datetime.datetime.now(),'optimizing clusters:')
# sys.stdout.flush()
# results = compare_partitions(gt_clusters,clusters)
# print('check split: gt vs cl',compare_partitions(gt_clusters,clusters))
# sys.stdout.flush()
# print('current ll is ', cl_opt_ll)
# sys.stdout.flush()
# last_p_opt_ll = cl_opt_ll

# print(results['nmi'])
# print(results['rand'])
# print(results['jaccard'])

#     while 1: # now optimize p_out and p_in until change will be less than EPS
#         print('#',datetime.datetime.now(),'optimizing alphas:')
#         sys.stdout.flush()
#         p_out, new_p_opt_ll = optimize_p(clusters, epidemics, p_in, p_out, 'p_out') #,dbg = True)
#         p_in,  new_p_opt_ll = optimize_p(clusters, epidemics, p_in, p_out, 'p_in') #,dbg = True)
#         print('new p_in',p_in,'new p_out',p_out, '\ncurrent ll is', new_p_opt_ll)
#         sys.stdout.flush()
#         if new_p_opt_ll - last_p_opt_ll < EPS:
#             last_p_opt_ll = new_p_opt_ll
#             break
#         last_p_opt_ll = new_p_opt_ll
#     if new_p_opt_ll - current_ll < EPS:
#         current_ll = new_p_opt_ll
#         break
#     current_ll = new_p_opt_ll

# print('done!\n\ntotal results:')
# print('p_in',p_in,'p_out',p_out, '\ncurrent ll is', new_p_opt_ll)
# print('check split: gt vs clusters',compare_partitions(gt_clusters,clusters))
# # print('clusters', clusters)
# # exit()

# out_fn = open(fn+'.alg12.start_from_netinf.clusters','w+')
# for n,c in clusters.items():
#     print(str(n)+"\t"+str(c), file=out_fn)
# out_fn.close()
# out_fn = open(fn+'.alg12.start_from_netinf.clusters.score.'+fn2,'w+')
# print(compare_partitions(gt_clusters,clusters), file=out_fn)
# out_fn.close()

fh = open('a12_results.tsv','w')
for node, cluster in sorted(clusters.items()):
    print('%d\t%d' % (node,cluster),file=fh)
fh.close()
