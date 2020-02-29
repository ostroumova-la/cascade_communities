# Cascade-based community detection

This is a supplementary code for the paper [When Less is More: Systematic Analysis of Cascade-based Community Detection](https://arxiv.org/pdf/2002.00840.pdf).

Supplementary directories:

**CommDiff-Package/**: the source code for R-CoDi and D-CoDi from the paper “Community Detection Using Diffusion Information” by Ramezani et al.

**CommunityWithoutNetworkLight/**: the source code for C-IC and C-Rate algorithms from the paper “Efficient methods for influence-based network-oblivious community detection” by Barbieri et al.

**network-inference-multitree/**: the source code for the MultiTree algorithm from the paper “Submodular inference of diffusion networks from multiple trees” by Gomez-Rodriguez et al.

**community_ext/**: [community detection library](https://github.com/altsoph/community_loglike) complementing the paper “Community detection through likelihood optimization: in search of a sound model” by Prokhorenkova et al. 

**benchmark/**: used to generate synthetic graphs according to the LFR model proposed in “Benchmark graphs for testing community detection algorithms” by Lancichinetti et al. 

Datasets directories:

**LFR_1000/**, **citeseer/**, **cora-small/**, **cora/**, **dolphins/**, **eu-core/**, **football/**, **karate/**, **newsgroup/**, **polblogs/**, **polbooks/**, **twitter/**.

Directories with results: **average_ranks/** and **average_results/** contain aggregated results over real-world datasets, **cascade_plots/** contains the distribution of cascade sizes.

Description of scripts:

**C-SI-BD.py**, **SI-BD.py**, **SIR.py** - to generate epidemics;

**base_algorithms.py**, **base_algorithms_twitter.py** - simple algorithms;

**baseline_barbieri.py**, **baseline_barbieri_twitter.py** - to run C-IC and C-Rate;

**baseline_cd.py**, **baseline_cd_twitter.py** - to run R-CoDi and D-CoDi;

**opt_algorithms.py**, **opt_algorithms_twitter.py** - GraphOpt and ClustOpt algorithms;

**cascade_plots.py**, **cascade_plots_twitter.py** - to generate data for plots;

**average_rank.py**, **average_results.py** to aggregate the results.

To reproduce the main experiments from the paper one can use the file **paper_experiments.tex**.



