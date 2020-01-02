import numpy as np
from collections import defaultdict

data = open("twitter/epidemics/30000", "r")
dist = defaultdict(lambda : 0)
for line in data:
    value = int(line.split()[3])
    dist[value] += 1
cum_dist = {}
sum = 0
for value in sorted(dist.keys(), reverse = True):
    sum += dist[value]
    cum_dist[value] = sum
dst1 = open("cascade_plots/dist_twitter", "w")
dst2 = open("cascade_plots/cum_dist_twitter", "w")
for value in sorted(dist.keys()):
    dst1.write(str(value)+"\t"+str(float(dist[value])/sum)+"\n")
    dst2.write(str(value)+"\t"+str(float(cum_dist[value])/sum)+"\n")