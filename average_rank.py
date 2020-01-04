import numpy as np

datasets = ["karate", "dolphins", "football", "polbooks", "polblogs", "eu-core", "newsgroup", "citeseer", "cora-small"]
epidemics = ["SI-BD", "SIR", "C-SI-BD"]
#algorithms = ["clique_0.0", "clique_-1.0", "clique_-2.0"]
algorithms = ["multitree", "wpath", "clique_0.0", "clique_-2.0", "ramezani", "cd.random", "cd.dani", "cwnl.cr_auto", "cwnl.cic_auto", "oracle", "a12_0.0"]
m = len(algorithms)
n = len(datasets)
l = 10
k = 15

for epidemic in epidemics:
    if epidemic == "C-SI-BD":
        algorithms = ["multitree", "wpath", "clique_0.0", "clique_-2.0", "ramezani", "cd.random", "cd.dani", "cwnl.cr_auto", "cwnl.cic_auto", "a12_0.0"]
        m = len(algorithms)
    results = np.zeros((m,n,l,k))
    i = 0
    for algorithm in algorithms:
        j = 0
        for dataset in datasets:
            result = np.loadtxt(dataset+"/results/"+epidemic+"_"+algorithm)[:10]
            results[i,j,:,:] = result[:,1:]
            j += 1
        i += 1
    x = result[:,0]
    print(results.shape)
    order = results.argsort(axis = 0)
    ranks = order.argsort(axis = 0)
    avg_ranks = np.mean(ranks, axis = 1)
    print avg_ranks.shape
    
    i = 0
    for algorithm in algorithms:   
        result = np.zeros((l,k+1))
        result[:,1:] = avg_ranks[i,:,:]
        result[:,0] = x
        np.savetxt("average_ranks/"+epidemic+"_"+algorithm, result)
        i += 1
