# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function
from collections import defaultdict, Counter
from math import log, exp, sqrt
from scipy.stats.stats import pearsonr, spearmanr
import numpy as np

def p2ev(p):
    v = []
    for i in range(len(p)-1):
        for j in range(i+1,len(p)):
            if p[i] == p[j]: 
                v.append(1.)
            else:
                v.append(0.)
    return v

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
    if eta_xy == 0.: return 0.,0.
    return sum_mi/sqrt(_eta(x)*_eta(y)),2.*sum_mi/(_eta(x)+_eta(y))

def compare_partitions(p1,p2,safe=True):
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
    assert not set(p1.keys())^set(p2.keys()) or not safe, 'You tried to compare partitions with different numbers of nodes. Consider using safe=False flag for compare_partitions() call.'
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
            cross_tab[0][0] += common * (common-1)
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
    nmis = _nmi(p1_vec,p2_vec)
    res = {
        'nmi': nmis[0],
        'nmi_arithm': nmis[1],
    }


    if (a00 + a01 + a10 + a11) == 0:
        res['rand'] = 1.
    else:
        res['rand'] = float(a00 + a11) / (a00 + a01 + a10 + a11)
    if (a01 + a10 + a00) == 0:
        res['jaccard'] = 1.
    else:
        res['jaccard'] = float(a00) / (a01 + a10 + a00)
    return res 


def get_cross_tab(p1_sets,p2_sets):
    cross_tab = [[0, 0], [0, 0]]
    for a1, s1 in enumerate(p1_sets):
        for a2, s2 in enumerate(p2_sets):
            common = len(s1 & s2)
            l1 = len(s1) - common
            l2 = len(s2) - common
            cross_tab[0][0] += common * (common-1)
            cross_tab[1][0] += common * l2
            cross_tab[0][1] += l1 * common
            cross_tab[1][1] += l1 * l2
    return cross_tab

def compare_partitions_metrics(p1,p2):
    res = dict()

    # рассчитываем кластера на подмножестве
    p1_unknown = set(p1.keys())-set(p2.keys())
    p2_unknown = set(p2.keys())-set(p1.keys())
    p1_sets = defaultdict(set)
    p2_sets = defaultdict(set)
    [p1_sets[item[1]].add(item[0]) for item in p1.items() if item[0] not in p1_unknown]
    [p2_sets[item[1]].add(item[0]) for item in p2.items() if item[0] not in p2_unknown]
    [[a00, a01], [a10, a11]] = get_cross_tab(p1_sets.values(),p2_sets.values())

    # print('orig',p1_sets,p2_sets)
    p1_vec = [item[1] for item in sorted(p1.items()) if item[0] not in p1_unknown]
    p2_vec = [item[1] for item in sorted(p2.items()) if item[0] not in p2_unknown]
    # print('vec',p1_vec,p2_vec,'\n')
    res['sub_nmi'],res['sub_nmi_arithm'] = _nmi(p1_vec,p2_vec)[:2]
    if not len(p1_sets):
        res['sub_fnmi'] = 0.
        res['sub_fnmi_arithm'] = 0.
    else:
        res['sub_fnmi'] = res['sub_nmi']*exp(-abs(len(p1_sets)-len(p2_sets))/len(p1_sets))
        res['sub_fnmi_arithm'] = res['sub_nmi_arithm']*exp(-abs(len(p1_sets)-len(p2_sets))/len(p1_sets))


    p1_edgevec = p2ev(dict(enumerate(p1_vec)))
    p2_edgevec = p2ev(dict(enumerate(p2_vec)))
    
    if np.var(p1_edgevec)==0 or np.var(p2_edgevec)==0:
        #print("problem")
        #print("p1 =", p1_edgevec)
        #print("p2 =", p2_edgevec)
        if p1_edgevec == p2_edgevec:
            res['sub_pearson'] = 1.
            #res['sub_spearman'] = 1.
        else:
            res['sub_pearson'] = 0.
            #res['sub_spearman'] = 0.
        #print("index = ", res['sub_pearson'])
    else:          
        res['sub_pearson'] = pearsonr(p1_edgevec,p2_edgevec)[0]
        #res['sub_spearman'] = spearmanr(p1_edgevec,p2_edgevec)[0]
        
    # res_dbg = dict()
    # res_dbg['ss_p1_ev'] = p1_edgevec
    # res_dbg['ss_p2_ev'] = p2_edgevec

    if (a00 + a01 + a10 + a11) == 0:
        res['sub_rand'] = 1.
    else:
        res['sub_rand'] = float(a00 + a11) / (a00 + a01 + a10 + a11)

    if (a01 + a10 + a00) == 0:
        res['sub_jaccard'] = 1.
    else:
        res['sub_jaccard'] = float(a00) / (a01 + a10 + a00)

    if (a11 + a11 + a10 + a01) == 0:
        res['sub_F-measure'] = 1.
    else:
        res['sub_F-measure'] = 2.*a11/(2.*a11+a10+a01)

    # рассчитываем кластера, где неизвестные в отдельных кластерах
    p1_sets_solo = p1_sets.copy()
    p2_sets_solo = p2_sets.copy()
    [p1_sets_solo[p1[item]].add(item) for item in p1_unknown]
    [p2_sets_solo[p2[item]].add(item) for item in p2_unknown]
    [p2_sets_solo['__solo2__%d' % len(p2_sets_solo)].add(item) for item in p1_unknown]
    [p1_sets_solo['__solo1__%d' % len(p1_sets_solo)].add(item) for item in p2_unknown]
    [[a00, a01], [a10, a11]] = get_cross_tab(p1_sets_solo.values(),p2_sets_solo.values())
    # print('solo',p1_sets_solo,p2_sets_solo)
    # res = dict()
    # if (a00 + a01 + a10 + a11) == 0:
    #     res['rand_fixed'] = 1.
    # else:
    #     res['rand_fixed'] = float(a00 + a11) / (a00 + a01 + a10 + a11)

    if (a01 + a10 + a00) == 0:
        res['jaccard_diff'] = 1.
    else:
        res['jaccard_diff'] = float(a00) / (a01 + a10 + a00)
    # return res 


    if (a11 + a11 + a10 + a01) == 0:
        res['F-measure_diff'] = 1.
    else:
        res['F-measure_diff'] = 2.*a11/(2.*a11+a10+a01)

    # рассчитываем кластера, где неизвестные в одном кластере
    p1_sets_single = p1_sets.copy()
    p2_sets_single = p2_sets.copy()
    [p1_sets_single[p1[item]].add(item) for item in p1_unknown]
    [p2_sets_single[p2[item]].add(item) for item in p2_unknown]
    [p2_sets_single['__single2__'].add(item) for item in p1_unknown]
    [p1_sets_single['__single1__'].add(item) for item in p2_unknown]
    [[a00, a01], [a10, a11]] = get_cross_tab(p1_sets_single.values(),p2_sets_single.values())

    p1_vec_single = p1_vec[:]
    p2_vec_single = p2_vec[:]
    p1_vec_single.extend( [p1[item] for item in p1_unknown] )
    if len(p2_vec):
        p2_vec_single.extend( [max(p2_vec)+1,]*len(p1_unknown)  )
    if len(p1_vec):
        p1_vec_single.extend( [max(p1_vec)+1,]*len(p2_unknown)  )
    p2_vec_single.extend( [p2[item] for item in p2_unknown] )
    # print('sing',p1_vec_single,p2_vec_single)
    res['nmi_fixed'],res['nmi_fixed_arithm'] = _nmi(p1_vec_single,p2_vec_single)[:2]
    res['fnmi_fixed'] = res['nmi_fixed']*exp(-abs(len(p1_sets_single)-len(p2_sets_single))/len(p1_sets_single))
    res['fnmi_fixed_arithm'] = res['nmi_fixed_arithm']*exp(-abs(len(p1_sets_single)-len(p2_sets_single))/len(p1_sets_single))
     # = nmis[1]

    p1_edgevec_v2 = p1_edgevec[:] # p2ev(dict(enumerate(p1_vec)))
    p2_edgevec_v2 = p2_edgevec[:] # p2ev(dict(enumerate(p1_vec)))
    for i in p1_unknown:
        for j in (set(p1.keys())-p1_unknown):
            if p1[i] == p1[j]:
                p1_edgevec_v2.append( 1. )
            else:
                p1_edgevec_v2.append( 0. )
            p2_edgevec_v2.append( 0.5 )
    for i in p2_unknown:
        for j in (set(p2.keys())-p2_unknown):
            if p2[i] == p2[j]:
                p2_edgevec_v2.append( 1. )
            else:
                p2_edgevec_v2.append( 0. )
            p1_edgevec_v2.append( 0.5 )

    if np.var(p1_edgevec_v2)==0 or np.var(p2_edgevec_v2)==0:
        #print("problem")
        #print("p1 = ", p1_edgevec)
        #print("p2 = ", p2_edgevec)
        if p1_edgevec_v2 == p2_edgevec_v2:
            res['pearson_v2'] = 1.
            #res['spearman_v2'] = 1.
        else:
            res['pearson_v2'] = 0.
            #res['spearman_v2'] = 0.
        #print("index = ", res['pearson_v2'])
    else:          
        res['pearson_v2'] = pearsonr(p1_edgevec_v2,p2_edgevec_v2)[0]
        #res['spearman_v2'] = spearmanr(p1_edgevec_v2,p2_edgevec_v2)[0]

    # print(p1_edgevec_v2)
    # print(p2_edgevec_v2)
    _m_0 = 0
    _m_05 = 0
    _m_1 = 0
    _l_0 = 0
    _l_05 = 0
    _l_1 = 0
    _n_1_1 = 0
    _n_05_1 = 0
    _n_1_05 = 0
    _n_05_05 = 0
    for i1,i2 in zip(p1_edgevec_v2,p2_edgevec_v2):
        if i1 == 0.:
            _m_0 += 1
        elif i1 == 1.:
            _m_1 += 1
        else:
            _m_05 += 1
        if i2 == 0.:
            _l_0 += 1
        elif i2 == 1.:
            _l_1 += 1
        else:
            _l_05 += 1
        if i1 == 1. and i2 == 1.:
            _n_1_1 += 1
        if i1 == .5 and i2 == 1.:
            _n_1_05 += 1
        if i1 == 1. and i2 == .5:
            _n_05_1 += 1
        if i1 == .5 and i2 == .5:
            _n_05_05 += 1
    _n = len(p1_edgevec_v2)
    # print(_m_0,_m_05,_m_1,_l_0,_l_05,_l_1,_n_1_1,_n_05_1,_n_1_05,_n_05_05,_n)
    __nom = _n*(_n_1_1 + .5*(_n_05_1+_n_1_05) + .25*_n_05_05) - (_l_1 + .5*_l_05)*(_m_1 + .5*_m_05)
    __denom = sqrt( (_m_0*_m_1 + .25*_m_0*_m_05 + .25*_m_1*_m_05)*(_l_0*_l_1 + .25*_l_0*_l_05 + .25*_l_1*_l_05)  )
    # print(__nom,__denom,__nom/__denom)

    p1_edgevec_v3 = p1_edgevec[:] # p2ev(dict(enumerate(p1_vec)))
    p2_edgevec_v3 = p2_edgevec[:] # p2ev(dict(enumerate(p1_vec)))
    for i in p1_unknown:
        for j in (set(p1.keys())-p1_unknown):
            if p1[i] == p1[j]:
                p1_edgevec_v3.append( 1. )
            else:
                p1_edgevec_v3.append( 0. )
            if j in p1_unknown:
                p2_edgevec_v3.append( 0.5 )
            else:
                p2_edgevec_v3.append( 0. )
    for i in p2_unknown:
        for j in (set(p2.keys())-p2_unknown):
            if p2[i] == p2[j]:
                p2_edgevec_v3.append( 1. )
            else:
                p2_edgevec_v3.append( 0. )
            if j in p2_unknown:
                p1_edgevec_v3.append( 0.5 )
            else:
                p1_edgevec_v3.append( 0. )
    if np.var(p1_edgevec_v3)==0 or np.var(p2_edgevec_v3)==0:
        #print("problem")
        #print("p1 = ", p1_edgevec_v3)
        #print("p2 = ", p2_edgevec_v3)
        if p1_edgevec_v3 == p2_edgevec_v3:
            res['pearson_v2_opt'] = 1.
            res['pearson_v3'] = 1.
            #res['spearman_v3'] = 1.
        else:
            res['pearson_v2_opt'] = 0.
            res['pearson_v3'] = 0.
            #res['spearman_v3'] = 0.
        #print("index = ", res['pearson_v3'])
    else:
        res['pearson_v2_opt'] = __nom/__denom
        res['pearson_v3'] = pearsonr(p1_edgevec_v3,p2_edgevec_v3)[0]
        #res['spearman_v3'] = spearmanr(p1_edgevec_v3,p2_edgevec_v3)[0]

    # return res_dbg
    return res 
