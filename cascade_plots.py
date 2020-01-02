import numpy as np
from collections import defaultdict

datasets = ["karate", "dolphins", "football", "polbooks", "polblogs", "eu-core", "newsgroup", "citeseer", "cora-small"]
len = {}
len[datasets[0]] = 10
len[datasets[1]] = 9
len[datasets[2]] = 8
len[datasets[3]] = 8

len[datasets[4]] = 4
len[datasets[5]] = 4
len[datasets[6]] = 4
len[datasets[7]] = 4
len[datasets[8]] = 4


epidemics = ["SI-BD-1", "C-SI-BD-1", "SIR-1"]

for epidemic in epidemics:
    for dataset in datasets:
        print(dataset+"/"+epidemic+"/"+str(len[dataset]))
        data = open(dataset+"/"+epidemic+"/"+str(len[dataset]), "r")
        dist = defaultdict(lambda : 0)
        for line in data:
            value = int(line.split()[3])
            dist[value] += 1
        cum_dist = {}
        sum = 0
        for value in sorted(dist.keys(), reverse = True):
            sum += dist[value]
            cum_dist[value] = sum
        dst1 = open("cascade_plots/dist_"+epidemic+"_"+dataset, "w")
        dst2 = open("cascade_plots/cum_dist_"+epidemic+"_"+dataset, "w")
        for value in sorted(dist.keys()):
            dst1.write(str(value)+"\t"+str(float(dist[value])/sum)+"\n")
            dst2.write(str(value)+"\t"+str(float(cum_dist[value])/sum)+"\n")