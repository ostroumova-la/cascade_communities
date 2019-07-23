package evaluation;

import java.util.ArrayList;
import java.util.HashMap;

import handlers.*;
import utils.*;
import beans.*;

public class CommunityStructureCalculator {

	private LinksHandler graph;
	private HashMap<Integer, Integer> node2CommunityMap;
	private ArrayList<Integer> communities[];
	private int nCommunities;

	public CommunityStructureCalculator() {
	}

	public CommunityStructureCalculator(LinksHandler network,
			CommunityAssignmentHandler assignment) {
		this.graph = network;
		this.node2CommunityMap = assignment.getVertex2Community();
		this.communities = assignment.getCommunities();
		this.nCommunities = assignment.getnCommunities();
	}
	
	public CommunityStructureCalculator(LinksHandler network, HashMap<Integer, Integer > node2Community,ArrayList<Integer> communities[]){
		this.graph=network;
		this.node2CommunityMap=node2Community;
		this.communities=communities;
		this.nCommunities=communities.length;
	}
	
	

	public LinksHandler getGraph() {
		return graph;
	}

	public void setGraph(LinksHandler graph) {
		this.graph = graph;
	}


	public int getnCommunities() {
		return nCommunities;
	}

	public void setnCommunities(int nCommunities) {
		this.nCommunities = nCommunities;
	}

	/*
	 * Modularity for undirected graph Reference: Eq 15 Fortunato. Q= \sum_{c}
	 * [lc/m - (dc/2m)^2 ] lc= total number of edges joining vertices of module
	 * c dc= the sum of the degrees of the vertices of c.
	 */
	public double computeModularity() {
		double lc[] = new double[nCommunities];
		double dc[] = new double[nCommunities];
		for (Edge e : graph.getUndirectedEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 == c2) {
				lc[c1]++;
			}
		}
		
		for (int nodeId : graph.getVertexArray()) {
			int c = node2CommunityMap.get(nodeId)-1;
			int degree=0;
			if(graph.getOutLinksForVertex(nodeId)!=null)
				degree = graph.getOutLinksForVertex(nodeId).size()+graph.getInlinksForVertex(nodeId).size();			
			dc[c] += degree;
		}
		double m = graph.getNUnDirectedEdges();
		double Q = 0.0;
		for (int c = 0; c < nCommunities; c++) {
			Q += lc[c] / m - Math.pow(dc[c] / (2 * m), 2);
		}
		return Q;
	}

	public double computeModularityDirectGraph() {
		int[] V = graph.getVertexArray();
		int[] k_in = new int[V.length];
		int[] k_out = new int[V.length];
		int m = graph.getNDirectedEdges();
		for (int i = 0; i < V.length; i++) {
			k_in[i] = graph.getInlinksForVertex(V[i]).size();
			k_out[i] = graph.getOutLinksForVertex(V[i]).size();
		}
		double Q = 0.0;
		for (int i = 0; i < V.length; i++) {
			for (int j = 0; j < V.length; j++) {
				int c_i = node2CommunityMap.get(V[i])-1;
				int c_j = node2CommunityMap.get(V[j])-1;
				if (c_i == c_j) {
					double A_ij = graph.existsEdge(V[i], V[j]) ? 1 : 0.0;
					Q += A_ij / m - k_out[i] * k_in[j] / Math.pow(m, 2);
				}
			}
		}
		return Q;
	}//computeModularityDirectGraph

	
	public double []getCommunitiesSize(){
		double[]ris=new double[nCommunities];
		
		for (int c = 0; c < nCommunities; c++) 
			ris[c]= communities[c].size();
			
		return ris;
	}
	
	
	/*
	 * Compute Conductance: Let S be the set of nodes in the cluster,
	 * f(S)=cS/(2*mS+cS) mS the number of edges in S cS, the number of edges on
	 * the boundary of S
	 */
	public double[] computeConductance() {
		double cond[] = new double[nCommunities];
		double cs[] = new double[nCommunities];
		double ms[] = new double[nCommunities];
		for (Edge e : graph.getUndirectedEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 == c2) {
				ms[c1]++;
			} else {
				cs[c1]++;
			//	cs[c2]++;
			}
		}
		for (int c = 0; c < nCommunities; c++) {
			cond[c] = cs[c] / (2 * ms[c] + cs[c]);
		}
		return cond;
	}//computeConductance

	
	/*
	 * Compute Conductance: Let S be the set of nodes in the cluster,
	 * f(S)=cS/(2*mS+cS) mS the number of edges in S cS, the number of edges on
	 * the boundary of S
	 */
	public double[] computeConductanceDirectedGraph() {
		double cond[] = new double[nCommunities];
		double cs[] = new double[nCommunities];
		double ms[] = new double[nCommunities];
		for (Edge e : graph.getEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 == c2) {
				ms[c1]++;
			} else {
				cs[c1]++;
				cs[c2]++;
			}
		}
		for (int c = 0; c < nCommunities; c++) {
			cond[c] = cs[c] / (ms[c] + cs[c]);
		}
		return cond;
	}//computeConductanceDirectedGraph
	
	
	/*
	 * Let S be the set of nodes in the cluster f(S)= (ms/(ns(ns-1)/2) where
	 * nS is the number of nodes in S mS the number of edges in S
	 */
	public double[] computeInternalDensity() {
		double id[] = new double[nCommunities];
		double ms[] = new double[nCommunities];
		for (Edge e : graph.getUndirectedEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 == c2) {
				ms[c1]++;
			}
		}
		for (int c = 0; c < nCommunities; c++) {
			int ns = communities[c].size();
			if (ms[c] == 0) {
				id[c] = 0.0;
			} else {
				id[c] = ms[c] / (ns * (ns - 1) / 2.0);
			}
		}
		return id;
	}//computeInternalDensity

	
	public double[] computeInternalDensityDirectedGraph() {
		double id[] = new double[nCommunities];
		double ms[] = new double[nCommunities];
		for (Edge e : graph.getEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 == c2) {
				ms[c1]++;
			}
		}
		for (int c = 0; c < nCommunities; c++) {
			int ns = communities[c].size();
			if (ms[c] == 0) {
				id[c] = 0.0;
			} else {
				id[c] = ms[c] / (ns * (ns - 1) );
			}
		}
		return id;
	}//computeInternalDensityDirectedGraph
	
	/*
	 * is the fraction of all possible edges leaving the cluster Let S be the
	 * set of nodes in the cluster f(S)= cs/(ns(n-ns) where nS is the number of
	 * nodes in S 
	 */
	public double[] computeCutRatio() {
		double[] cr = new double[nCommunities];
		double cs[] = new double[nCommunities];
		for (Edge e : graph.getUndirectedEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 != c2) {
				cs[c1]++;
			//	cs[c2]++;
			}
		}
		int n = graph.getNVertices();
		for (int c = 0; c < nCommunities; c++) {
			int ns = communities[c].size();
			cr[c] = cs[c] / (ns * (n - ns));
		}
		return cr;
	}
	
	
	public double[] computeCutRatioDirectedGraph() {
		double[] cr = new double[nCommunities];
		double cs[] = new double[nCommunities];
		for (Edge e : graph.getEdges()) {			
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 != c2) {
				cs[c1]++;
				cs[c2]++;
			}
		}
		int n = graph.getNVertices();
		for (int c = 0; c < nCommunities; c++) {
			int ns = communities[c].size();
			cr[c] = cs[c] / (2*ns * (n - ns));
		}
		return cr;
	}//
	
	
	public double[] computeNormalizedCutRatio() {
		double[] ncr = new double[nCommunities];
		double cs[] = new double[nCommunities];
		double ms[] = new double[nCommunities];
		for (Edge e : graph.getUndirectedEdges()) {
			int c1 = node2CommunityMap.get(e.source)-1;
			int c2 = node2CommunityMap.get(e.destination)-1;
			if (c1 != c2) {
				cs[c1]++;
			//	cs[c2]++;
			}
			else
				ms[c1]++;
			
		}
		int m = graph.getNUnDirectedEdges();
		for (int c = 0; c < nCommunities; c++) {
			ncr[c] = cs[c] / (2*ms[c] + cs[c]) +  cs[c]/(2*(m-ms[c])+cs[c]  )   ;
		}
		return ncr;
	}//computeNormalizedCutRatio
	
	/*
	Entropy= - 1/(2K^2)\sum_k \sum_k' P(k,k') \log �P(k,k') + ( 1-P(k,k') ) \log �( 1-P(k,k') )
	P(k,k')= | �(u,v) , u \in k, v \in k' � | / � | k | x |k'|
	*/
	public double computeEntropy(){
		double entropy=0.0;
		int counts_k_kprime[][]=new int[nCommunities][nCommunities];
		double ns[]=new double[nCommunities];
		for (Edge e : graph.getEdges()) {
			
			Integer c1 = node2CommunityMap.get(e.source);
			Integer c2 = node2CommunityMap.get(e.destination);
			
				counts_k_kprime[c1-1][c2-1]++;
				
		}
		for(int c=0;c<nCommunities;c++)
			ns[c]=communities[c].size();
		
		for(int k=0;k<nCommunities;k++){
			for(int k1=0;k1<nCommunities;k1++){
				double P=(double)counts_k_kprime[k][k1]/(ns[k]*ns[k1]);
				if(P==0)
					P=Math.pow(10,-8);
				if(P==1)
					P=0.99999;
				entropy+= P*Math.log(P) + (1-P)*Math.log(1-P);
			}
			
		}
	
		return -entropy/(double)(2*nCommunities*nCommunities); 
	}
	
	
	public void evaluateAll(){
		System.out.println(" - - - - - - Results (K = "
				+ nCommunities+ ") - - - - - - - ");
		System.out.println("Modularity:");
		printForExcel(computeModularityDirectGraph());
		System.out.println("Conductance:");
		double[] cond = computeConductanceDirectedGraph();
		System.out.println("Min\t"+ArrayUtilities.findMin(cond)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(cond)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(cond));
		System.out.println("Harmonic mean\t"+ArrayUtilities.HarmonicMean(cond));
		System.out.println("Median \t"+ArrayUtilities.median(cond));
		System.out.println("Internal Density:");
		double[] id = computeInternalDensityDirectedGraph();
		System.out.println("Min\t"+ArrayUtilities.findMin(id)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(id)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(id));
		System.out.println("Harmonic mean\t"+ArrayUtilities.HarmonicMean(id));
		System.out.println("Median\t"+ArrayUtilities.median(id));

		System.out.println("Cut Ratio:");
		double[] cr = computeCutRatioDirectedGraph();
		System.out.println("Min\t"+ArrayUtilities.findMin(cr)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(cr)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(cr));
		System.out.println("Harmonic mean\t"+ArrayUtilities.HarmonicMean(cr));
		System.out.println("Median\t"+ArrayUtilities.median(cr));

		double entropy=computeEntropy();
		System.out.println("Entropy");
		System.out.println(entropy);
		System.out.println(" - - - - - - - - - - - - - - - - - - - - -");
	}

	public static void main(String[] args) throws Exception {
		System.out.println(" - - - - Community Structure Evaluator- - - - ");
		
		if(args.length==0){
			printUsage();
			return;	
		}
		
		
		String networkPath=null;
		String communityMembership=null;
		boolean directed=false;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("--help")) {
				printUsage();
				return;
			}
			
			if (args[i].equals("-n")) {
				networkPath = args[i + 1];
				i++;
			}
			
			
			if (args[i].equals("-c")) {
				communityMembership = args[i + 1];
				i++;
			}

			if (args[i].equals("-d")) {
				String s=args[i+1];
				if(s.equalsIgnoreCase("y")||s.equalsIgnoreCase("yes")||s.equalsIgnoreCase("true")  )
					directed=true;
				i++;
			}	
			
			
		}// for each args
	
		
		LinksHandler network = new LinksHandler();
		network.read(networkPath);
		network.printInfo();

		HashMap<Integer, Integer>assignments=Evaluation_Utils.readAssignments(communityMembership);		
		ArrayList<Integer>[] communities=Evaluation_Utils.readCommunities(communityMembership);
		
		System.out.println("Assignments\t"+assignments.size());
		
		CommunityStructureCalculator csc = new CommunityStructureCalculator(
				network, assignments,communities);
		
		

		System.out.println(" - - - - - - Results (K = "+ communities.length + ") - - - - - - - ");
		
		
		double modularity=0.0;
		
		double[] conductance;
		double []internalDensity;
		double []cutratio;
		
		if(directed){
			modularity=csc.computeModularityDirectGraph();
			conductance=csc.computeConductanceDirectedGraph();
			internalDensity=csc.computeInternalDensityDirectedGraph();
			cutratio=csc.computeCutRatioDirectedGraph();
		}
		else{
			modularity=csc.computeModularity();
			conductance=csc.computeConductance();
			internalDensity=csc.computeInternalDensity();
			cutratio=csc.computeCutRatio();
		}
		
		double entropy=csc.computeEntropy();
		
		double sizes[]=csc.getCommunitiesSize();
		
		System.out.println(" - - - Modularity - - - ");
		printForExcel(modularity);
		System.out.println();
		System.out.println(" - - - Conductance - - - - ");
		System.out.println("Min");
		printForExcel(ArrayUtilities.findMin(conductance)[1]);
		System.out.println("Max");
		printForExcel(ArrayUtilities.findMax(conductance)[1]);
		System.out.println("Avg");
		printForExcel(ArrayUtilities.avg(conductance));
		System.out.println("harmonic mean");
		printForExcel(ArrayUtilities.HarmonicMean(conductance));
		System.out.println("Median mean");
		printForExcel(ArrayUtilities.median(conductance));
		System.out.println();
		System.out.println("- - - Internal Density - - - ");
		System.out.println("Min");
		printForExcel(ArrayUtilities.findMin(internalDensity)[1]);
		System.out.println("Max");
		printForExcel(ArrayUtilities.findMax(internalDensity)[1]);
		System.out.println("Avg");
		printForExcel(ArrayUtilities.avg(internalDensity));
		System.out.println("harmonic mean");
		printForExcel(ArrayUtilities.HarmonicMean(internalDensity));
		System.out.println("Median mean");
		printForExcel(ArrayUtilities.median(internalDensity));


		System.out.println("- - - Cut Ratio- - - ");
		System.out.println("Min");
		printForExcel(ArrayUtilities.findMin(cutratio)[1]);
		System.out.println("Max");
		printForExcel(ArrayUtilities.findMax(cutratio)[1]);
		System.out.println("Avg");
		printForExcel(ArrayUtilities.avg(cutratio));
		System.out.println("harmonic mean");
		printForExcel(ArrayUtilities.HarmonicMean(cutratio));
		System.out.println("Median mean");
		printForExcel(ArrayUtilities.median(cutratio));
		System.out.println();
		
		System.out.println("- - - Sizes- - - ");
		System.out.println("Min");
		printForExcel(ArrayUtilities.findMin(sizes)[1]);
		System.out.println("Max");
		printForExcel(ArrayUtilities.findMax(sizes)[1]);
		System.out.println("Avg");
		printForExcel(ArrayUtilities.avg(sizes));
		System.out.println("harmonic mean");
		printForExcel(ArrayUtilities.HarmonicMean(sizes));
		System.out.println("Median mean");
		printForExcel(ArrayUtilities.median(sizes));
		System.out.println();
		
		System.out.println("Entropy\t"+entropy);

		
	}

	private static void printForExcel(double v) {
		System.out.println(String.valueOf(v).replace('.', ','));
	}

	private static void printUsage() {
		System.out.println("-n <networkFile> -c <communityFile>  -d <directed(yes|no)>");
	}
}// CommunityStructureCalculator