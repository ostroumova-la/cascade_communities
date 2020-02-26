# Cascade-based community detection

This is a supplementary code for the paper [When Less is More: Systematic Analysis of Cascade-based Community Detection](https://arxiv.org/pdf/2002.00840.pdf).

Supplementary directories:

**CommDiff-Package/**: the source code for R-CoDi and D-CoDi from the paper “Community Detection Using Diffusion Information” by Ramezani et al.

**CommunityWithoutNetworkLight/**: the source code for C-IC and C-Rate algorithms from the paper “Efficient methods for influence-based network-oblivious community detection” by Barbieri et al.

**community_ext/**: [community detection library](https://github.com/altsoph/community_loglike) complementing the paper “Community detection through likelihood optimization: in search of a sound model” by Prokhorenkova et al. 

Datasets directories:

**LFR_1000/**, **citeseer/**, **cora-small/**, **cora/**, **dolphins/**, **eu-core/**, **football/**, **karate/**, **newsgroup/**, **polblogs/**, **polbooks/**, **twitter/**.


Directories correspond to different datasets. The following assumes that we are in the corresponding dirrectory.

1. Find good parameters for synthetic epidemics for a givent dataset. To do this, run: 

python3 ../SI-BD.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 0.15 1 100000 > epidemics
cut -f 4 epidemics | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We want the average size of the obtained cascades to be 2, this means that the first number in the output is close to 1.

python3 ../SIR.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 12 100000 > epidemics
cut -f 4 epidemics | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We want the average size of the obtained cascades to be 2, this means that the first number in the output is close to 1.

python3 ../C-SI-BD.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 0.09 0.009 1 100000 > epidemics
cut -f 4 epidemics | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We take $p_{in} = 10 p_{out}$ and choose $p_{out}$ such that the number of cascades consisting of one node is about 20\%, this means that the third number in the output is close to 0.8

2. Generate epidemics

Use files "generate_SI-BD", "generate_C-SI-BD" and "generate_SIR" to get commands to run for a given dataset. This will produce 5 epidemic samples for each epidemic type. In each case, we generate slices.

Do not forget that we need enough cascades to make slices.

3. Run algorithms

Use file run_algorithms for commands to run

