# Learning clusters through information diffusion

Directories correspond to different datasets. The following assumes that we are in the corresponding dirrectory.

1. Find good parameters for synthetic epidemics for a givent dataset. To do this, run: 

python3 ../SI-BD.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 0.15 1 100000 > results
cut -f 4 results | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We want the average size of the obtained cascades to be 2, this means that the first number in the output is close to 1.

python3 ../SIR.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 12 100000 > results
cut -f 4 results | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We want the average size of the obtained cascades to be 2, this means that the first number in the output is close to 1.

python3 ../C-SI-BD.py ../tune_parameter.yaml graph/graph.clusters graph/graph.edges 0.09 0.009 1 100000 > results
cut -f 4 results | sort | uniq -c | sort -n -k2 > freq.txt
python3 ../make_cum_freq.py cum_freq.txt

We take $p_{in} = 10 p_{out}$ and choose $p_{out}$ such that the number of cascades consisting of one node is about 20\%, this means that the third number in the output is close to 0.8

2. Generate epidemics

Use files "generate_SI-BD", "generate_C-SI-BD" and "generate_SIR" to get commands to run for a given dataset. This will produce 5 epidemic samples for each epidemic type. In each case, we generate slices

3. To run base algorithms

