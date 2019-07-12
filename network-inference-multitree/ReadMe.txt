========================================================================
    Submodular Inference of Diffusion Networks from Multiple Trees
========================================================================

Diffusion and propagation of information, influence and diseases take place
over increasingly larger networks. We observe when a node copies
information, makes a decision or becomes infected but networks are often
hidden or unobserved. Since networks are highly dynamic, changing and
growing rapidly, we only observe a relatively small set of cascades before a
network changes significantly. Scalable network inference based on a small
cascade set is then necessary for understanding the rapidly evolving
dynamics that govern diffusion.

We have developed a scalable approximation algorithm with provable
near-optimal performance based on submodular maximization which achieves a
high accuracy in such scenario.

For more information about the procedure see:
  Submodular Inference of Diffusion Networks from Multiple Trees
  Manuel Gomez-Rodriguez, Bernhard Schölkopf
  http://www.stanford.edu/~manuelgr/network-inference-multitree/
  
In order to compile on MacOS: 'make' OR 'make opt'.
In order to compile in Linux: 'make linux' OR 'make opt_linux'. 
The code should also work in Windows but you will need to edit the Makefile.
'make opt' and 'make opt_linux' compile the optimized (fast) version of the code.

/////////////////////////////////////////////////////////////////////////////
Parameters:

   -i:Input cascades (one file) (default:'example-cascades.txt')
   -n:Input ground-truth network (one file) (default:'example-network.txt')
   -o:Output file name(s) prefix (default:'network')
   -e:Number of iterations (default:'5')
   -a:Alpha for transmission model (default:1)
   -d:Delta for power law model (default:1)
   -nc:Number of cascades to use (-1 indicates all)   
   -m:Transmission model (0: exponential, 1:powerlaw, 2:rayleigh) (default: 0)
   -s:How much additional files to create?
    1:info about each edge, 2:objective function value (+upper bound), 3:Precision-recall plot, 4:ROC plot, 5:all-additional-files (default:1)
 (default:1)
    
Generally -s:1 is the fastest (computes the least additional stuff), while -s:3 
takes longest to run but calculates all the additional stuff.

/////////////////////////////////////////////////////////////////////////////
Usage:

Infer the network given a text file with cascades (nodes and timestamps):

network-inference-multitree -i:example_cascades.txt

/////////////////////////////////////////////////////////////////////////////
Format input cascades:

The cascades input file should have two blocks separated by a blank line. 
- A first block with a line per node. The format of every line is <id>,<name>
- A second block with a line per cascade. The format of every line is <id>,<timestamp>,<id>,<timestamp>,<id>,<timestamp>...

A demo input file can be found under the name example-cascades.txt.
/////////////////////////////////////////////////////////////////////////////
Format gound truth:

The ground truth input file should have two blocks separated by a blank line
- A first block with a line per node. The format of every line is <id>,<name>
- A second block with a line per directed edge. The format of every line is <id1>,<id2>,<alpha value>

/////////////////////////////////////////////////////////////////////////////
Additional Tool:

In addition, generate_nets is also provided. It allows to build Kronecker and Forest-Fire networks and generate cascades
with exponential, powerlaw and rayleigh transmission models. Please, run without any argument to see how to use them.