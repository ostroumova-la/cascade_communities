import numpy as np

datasets = ["karate", "dolphins", "football", "polbooks", "polblogs", "eu-core", "newsgroup", "citeseer", "cora-small"]
epidemics = ["SI-BD", "C-SI-BD", "SIR"]
#algorithms = ["clique_0.0", "clique_-1.0", "clique_-2.0"]
algorithms = ["oracle", "multitree", "path", "wpath", "clique_0.0", "clique_-1.0", "clique_-2.0", "ramezani", "cd.random", "cd.dani", "cwnl.cr_auto", "cwnl.cic_auto", "oracle", "a12_0.0", "a12_-1.0", "a12_-2.0"]
num = len(datasets)

for epidemic in epidemics:
    for algorithm in algorithms:
        i = 0
        for dataset in datasets:
            print(epidemic, algorithm, dataset)
            result = np.loadtxt(dataset+"/results/"+epidemic+"_"+algorithm)[:10]
            if i == 0:
                results = result
                i += 1
            else:
                results += result
        np.savetxt("average_results/"+epidemic+"_"+algorithm, results/num)