package cicm;

import handlers.ActivationsHandler;
import handlers.LinksHandler;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import evaluation.ClusteringEvaluator;
import evaluation.CommunityStructureCalculator;
import evaluation.EvaluateMutualInformation;
import evaluation.Evaluation_Utils;
import util.ArrayUtilities;
import util.Randoms;
import util.IOManager;
import utils.Weka_Utils;
import beans.Activation;

public class CommunityIC_Inference_Alt {
	
	private int nVertices;
	private Integer[] vertexSet;
	private Int2IntOpenHashMap vertexId2Index;
	private Int2IntOpenHashMap traceId2Index;

	private int nTraces;
	private Integer[] traceSet;
	
	private LinksHandler network;

	static double min_value=0.00000001;
	static double threshold=0.00000001;

	private int save_step=10;

	int nCommunities=-1;
	int nMaxIterations=100;
	
	CommunityIC_Model temp;
	
	double gamma[][];
	double priors[];
	double influenceProbabilities[][];

	private HashMap<Integer, Integer> community_assignments_ground;
	private ArrayList<Integer>[]communities_ground;

	private HashMap<Integer, Integer> assignments;
	private ArrayList<Integer>[]communities;

	
	private HashMap<Integer, Double[]> cache_activation_prob;
	int[]n_adoption_users;
	Randoms rand=new Randoms();

	private ArrayList<Activation>[] traces;
	
	static double inf_prob_internal_min=0.01;
	static double inf_prob_internal_max=0.2;

	static double inf_prob_external_max=0.001;
	static double inf_prob_external_min=0.02;

	
	private String outputFile;
	
	public void build(ActivationsHandler a,int k,int nMaxIt, String outputFile) throws Exception{
		
		System.out.println("Community IC Inference with "+k+" Communities");
		
		this.nMaxIterations=nMaxIt;
		this.nCommunities=k;
		this.outputFile=outputFile;

		
		this.nVertices = a.getNUsers();
		this.vertexSet = a.getUserSet().toArray(new Integer[nVertices]);
		
		this.vertexId2Index = a.getUserIndex();
		this.traceId2Index=a.getItemIndex();
		
		a.getItemIndex();
		this.nTraces = a.getNItems();
		this.traceSet = a.getItemSet().toArray(new Integer[nTraces]);
		this.assignments=new HashMap<Integer, Integer>();
		this.communities=new ArrayList[nCommunities];
		
		this.traces = new ArrayList[nTraces];
		for (int i = 0; i < nTraces; i++) {
			traces[i] = a.getActionsForItem(traceSet[i]);
			Collections.sort(traces[i]);
		}
		init();
		iterate();		
	}//build
	
	
	private void init(){
		System.out.println("Init model...");
		this.priors = new double[nVertices];
		Arrays.fill(priors, min_value);
	
		this.influenceProbabilities=new double[nVertices][nCommunities];
		this.gamma=new double[nVertices][nCommunities];
		
		for (int u = 0; u < nVertices; u++) {
			for (int k = 0; k < nCommunities; k++) {
				gamma[u][k] = Math.max(min_value, rand.nextDouble());	
			}
			Weka_Utils.normalize(gamma[u]);
		}
		
		n_adoption_users=new int[nVertices];
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				n_adoption_users[u]++;
			}
		}
		
		
		// init values!!!
		for (int u = 0; u < nVertices; u++){
			int z_u=Weka_Utils.maxIndex(gamma[u]);
			for (int c = 0; c < nCommunities; c++)
				if(z_u==c){
					influenceProbabilities[u][c]=rand.nextUniform(inf_prob_internal_min, inf_prob_internal_max);
				}
				else{
					influenceProbabilities[u][c]=rand.nextUniform(inf_prob_external_min, inf_prob_external_max);
				}
		}
		
		cache_activation_prob=new HashMap<Integer, Double[]>();
		
		
		int activation_index=0;
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			
			double log_A_i_u_k[]=new double[nCommunities];
			int activation_trace_index=0;
			
			for(Activation a: traces[traceIndex]){
				
				int u=vertexId2Index.get(a.userId);
				
				Double a_i_u[]=new Double[nCommunities];
				Arrays.fill(a_i_u, 0.0);
				for(int k=0;k<nCommunities;k++){
					if(activation_trace_index>0)
						a_i_u[k]=Math.exp(log_A_i_u_k[k]);
					log_A_i_u_k[k]+=Math.log(1-influenceProbabilities[u][k]);
				}
				
				if(cache_activation_prob.containsKey(activation_index))
					throw new RuntimeException("Duplicate hash");
				
				cache_activation_prob.put(activation_index,a_i_u);
				
				activation_index++;
				activation_trace_index++;
			}//for each activation on the trace
				
		}//for each trace
		
		M_Step();
			
		System.out.println("------------------------------------------------");
		
		updateAssigments();
		
		if(community_assignments_ground!=null){
			ClusteringEvaluator.evaluate(nCommunities, community_assignments_ground, assignments);
			EvaluateMutualInformation.evaluate(communities, communities_ground);
		}
		
		if(network!=null){
			CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
			csc.evaluateAll();
		}
		
	}//init 
	
	
	private int updateAssigments(){
		communities=new ArrayList[nCommunities];
		for(int c=0;c<nCommunities;c++)
			communities[c]=new ArrayList<Integer>();
		
		int countSwitch=0;
		for(int u=0;u<nVertices;u++){
			int z_u=Weka_Utils.maxIndex(gamma[u])+1;
			Integer old_z=assignments.get(vertexSet[u]);
			if(old_z!=null){
				if(old_z!=z_u)
					countSwitch++;
			}
			assignments.put(vertexSet[u], z_u);	
			
			communities[z_u-1].add(vertexSet[u]);
			
		}//for each user	
		
		int n_not_empty_Communities=0;
		for(int c=0;c<nCommunities;c++)
			if(communities[c].size()!=0)
				n_not_empty_Communities++;
		
		if(n_not_empty_Communities<nCommunities){
			ArrayList<Integer>comm2[]=new ArrayList[n_not_empty_Communities];
			int counter=0;
			for(int c=0;c<nCommunities;c++)
				if(communities[c].size()!=0){
					comm2[counter]=communities[c];
					counter++;
				}
			communities=comm2;
			
			assignments.clear();
			
			for(int c=0;c<communities.length;c++)
				for(int userId:communities[c])
					assignments.put(userId,c+1);	
		}

		return countSwitch;
				
	}//updateAssigments
	
	private void iterate(){
		
		
		System.out.println("Learning phase:\tstarting");
		System.out.println("#Iteration\tChanges\tTime\tLLk");
	
		int saveIteration = 0;
		long initTime = System.currentTimeMillis();
			
		for (int iterationsDone = 1; iterationsDone <= nMaxIterations; iterationsDone++) {
			
			
			double llk=E_Step();
			
			int changes=updateAssigments();

			M_Step();
			
			
			System.out.println(iterationsDone + "\t"+changes+"\t"+(System.currentTimeMillis() - initTime)+"\t"+llk);

			
			saveIteration++;
			if (saveIteration == save_step) {
				saveIteration = 0;
				
				if(community_assignments_ground!=null){
					ClusteringEvaluator.evaluate(nCommunities, community_assignments_ground, assignments);
					EvaluateMutualInformation.evaluate(communities, communities_ground);
				}
				
				if(network!=null){
					CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
					csc.evaluateAll();
				}
				
			}
		}// for each iteration

		long learningTime = System.currentTimeMillis() - initTime;
		System.out.println("Learning Phase: DONE  ("
				+ ((double) learningTime / 1000) + " secs)\nWriting the model");

		
		temp = buildModel();
		try {
			IOManager.writeObjectOnFile(temp, outputFile);
			File f=new File(outputFile + "TMP");
			f.delete();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		if(community_assignments_ground!=null){
			ClusteringEvaluator.evaluate(nCommunities, community_assignments_ground, assignments);
			EvaluateMutualInformation.evaluate(communities, communities_ground);
		}
		
		if(network!=null){
			CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
			csc.evaluateAll();
		}
		
		
		
		// Write communities
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(outputFile
					+ "Communities"));
			pw.println("Vertex\tCommunity");
			for (int u = 0; u < nVertices; u++) {
				pw.println("" + vertexSet[u] + "\t"
						+ Weka_Utils.maxIndex(gamma[u]));
			}
			pw.flush();
			pw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.print("DONE");
		
	}//iterate

	
	
	private double E_Step(){
		
		double llk=0.0;
		
		cache_activation_prob.clear();
		
		double log_pi[]=new double[nCommunities];
		for(int k=0;k<nCommunities;k++)
			log_pi[k]=Math.log(priors[k]);
		
	
	
		double A_u_k[][]=new double[nVertices][nCommunities];
		double B_u_k[][]= new double[nVertices][nCommunities];
		double B_i_k[][]= new double[nTraces][nCommunities];
		double B_k[]= new double[nCommunities];

		int activation_index=0;
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			double log_A_i_u_k[]=new double[nCommunities];

			int activation_trace_index=0;
			
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
		
				Double A_i_u_k[]=new Double[nCommunities];
				Arrays.fill(A_i_u_k, 0.0);
				
				for(int k=0;k<nCommunities;k++){
								
					if(activation_trace_index>0){
						A_u_k[u][k]+= Weka_Utils.log1mexp(log_A_i_u_k[k]);
						//A_u_k[u][k]+= Math.log(1-Math.exp(log_A_i_u_k[k]));
						A_i_u_k[k]=	Math.exp(log_A_i_u_k[k]);
					}	
					
					log_A_i_u_k[k]+= Math.log(1-influenceProbabilities[u][k]);
					B_i_k[traceIndex][k]+=Math.log(1-influenceProbabilities[u][k]);
					
					
				}//for each community
				
				cache_activation_prob.put(activation_index,A_i_u_k);
				activation_trace_index++;
				activation_index++;
			}//for each activation on the trace
			

		}//for each trace
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			
			for(int k=0;k<nCommunities;k++){
				B_k[k]+=B_i_k[traceIndex][k];
				
			}//B_k
			
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				for(int k=0;k<nCommunities;k++){
					B_u_k[u][k]+=B_i_k[traceIndex][k];
				}
			}//B_u_k
			
		}//B_k & B_u_k
		
		
			
		
		
		for(int u=0;u<nVertices;u++){
			double llk_u=0.0;
			double []log_gammma_u=new double[nCommunities];
			for(int k=0;k<nCommunities;k++){
				log_gammma_u[k]=A_u_k[u][k]+B_k[k]-B_u_k[u][k]+log_pi[k];
				llk_u+=gamma[u][k]* (log_gammma_u[k]) ;
			}
			try{
			gamma[u]=Weka_Utils.logs2probs(log_gammma_u);	
			}catch(Exception e ){
				ArrayUtilities.print(A_u_k[u]);
				ArrayUtilities.print(B_k);
				ArrayUtilities.print(B_u_k[u]);
				ArrayUtilities.print(log_pi);

				e.printStackTrace();
				System.exit(-1);
				
			}
			
			llk+=llk_u;
		}//for each vertex
		
		return llk;
	}//E_Step
	
	private void M_Step(){
		
		double Gamma[]=new double[nCommunities];
		double Gamma_i_k[][]=new double[nTraces][nCommunities];
		double Gamma_u_k[][]=new double[nVertices][nCommunities];

		double C_u_k[][]=new double[nVertices][nCommunities];
		double C_i_k[][]=new double[nTraces][nCommunities];
		double D_u_k[][]=new double[nVertices][nCommunities];
		
		
		double priors_new[]=new double[nCommunities];
		Arrays.fill(priors_new, min_value);
	
		for(int u=0;u<nVertices;u++){
			for(int k=0;k<nCommunities;k++)
				Gamma[k]+=gamma[u][k];
		}//Gamma
		
		
		
		int activation_index=0;
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			int trace_activation_index=0;
			
			double[]Gamma_i_u_k=new double[nCommunities];
			double[]C_i_u_k=new double[nCommunities];
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				for(int k=0;k<nCommunities;k++){
					Gamma_i_k[traceIndex][k]+=gamma[u][k];
					Gamma_i_u_k[k]+=gamma[u][k];
					Gamma_u_k[u][k]+=Gamma_i_u_k[k];
						
			
				//	if(trace_activation_index>0){	
						Double A_i_u_k[]=cache_activation_prob.get(activation_index);

						C_i_k[traceIndex][k]+=gamma[u][k]/(1-A_i_u_k[k]);	
						C_i_u_k[k]+=gamma[u][k]/(1-A_i_u_k[k]);
						D_u_k[u][k]+=C_i_u_k[k];
				//	}
				}//for each k
				
				trace_activation_index++;
				activation_index++;
			}//for each activation in the trace
			
		}//for each trace
	
		
		//C_u_k
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				for(int k=0;k<nCommunities;k++)
					C_u_k[u][k]+=C_i_k[traceIndex][k];
			}
		}
		
		for(int u=0;u<nVertices;u++){
			for(int k=0;k<nCommunities;k++){
				double num=C_u_k[u][k]-D_u_k[u][k];
				double den=(double)n_adoption_users[u]*Gamma[k]-Gamma_u_k[u][k];
				
				double inf_prob_u_k=0.0;
				if(num!=0.0)
					inf_prob_u_k=influenceProbabilities[u][k] * (num/den);
				
				
				if(inf_prob_u_k<min_value)
					inf_prob_u_k=min_value;
				
				if(inf_prob_u_k>1.0)
					throw new RuntimeException("Inf prob bak\t" +influenceProbabilities[u][k]+" new\t"+ inf_prob_u_k+" "+num+" "+den);
				
				if(inf_prob_u_k==1.0){
					System.err.println("Warning inf_prob_u_k=1.0 ");
					inf_prob_u_k=inf_prob_internal_max;
				}
				
				influenceProbabilities[u][k]=inf_prob_u_k;
			}
		}
		
	}//M-step
	
	
	
	private CommunityIC_Model buildModel(){
		CommunityIC_Model model=new CommunityIC_Model();
		model.setGamma(gamma);
		model.setInfluence_weights(influenceProbabilities);
		model.setnCommunities(nCommunities);
		model.setUsers(vertexSet);
		model.setUserIdToIndex(vertexId2Index);
		model.setPriors(priors);
		return model;
	}//buildModel
	
	
	public static void main(String[] args) throws Exception {
		System.out.println(CommunityIC_Inference_Alt.class.getSimpleName()
				+ " running on " + System.getProperty("os.name") + " OS ("
				+ System.getProperty("os.arch") + ")");
		
		
	/*	String network="s1";
		String propagationModel="NR";
		int iterations=100;
		String fold="fold1";
		
		String folder = "resources/datasets/synthetic/"+network+"/"+propagationModel+"/"+fold+"/";
		args = new String[] { "-a", folder + "actionLog", "-o", folder + "C-IC_Model_", "-k", "9",
				"-maxIt", ""+iterations, "-g", "resources/datasets/synthetic/"+network+"/community.dat" };
		*/
	
		String actionsFile = null;
		int nCommunities = 3;
		int nMaxIteration = 20;
		String groundTruthFile=null;
		String confFile=null;
		// long deltaT=1000;
		String outputFile = "C-IC_Model";
		String linksFile=null;

		if (args.length == 0) {
			printUsage();
			return;
		}

		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("--help")) {
				printUsage();
				return;
			}

			if (args[i].equals("-a")) {
				actionsFile = args[i + 1];
				i++;
			}


			if (args[i].equals("-k")) {
				nCommunities = Integer.parseInt(args[i + 1]);
				i++;
			}

			if (args[i].equals("-o")) {
				outputFile = args[i + 1];
				i++;
			}

			if (args[i].equals("-maxIt")) {
				nMaxIteration = Integer.parseInt(args[i + 1]);
				i++;
			}

			if (args[i].equals("-g")) {
				groundTruthFile =args[i + 1];
				i++;
			}
			
			if (args[i].equals("-c")) {
				confFile = args[i + 1];
				i++;
			}
			
			if (args[i].equals("-l")) {
				linksFile =args[i + 1];
				i++;
			}




		}// for each arg

		ActivationsHandler p = new ActivationsHandler();
		p.read(actionsFile, confFile);
		p.printInfo();

		
		
		CommunityIC_Inference_Alt inf=new CommunityIC_Inference_Alt();

		if(groundTruthFile!=null){
			inf.community_assignments_ground=Evaluation_Utils.readAssignments(groundTruthFile);
			inf.communities_ground=Evaluation_Utils.readCommunities(groundTruthFile);
		}
		
		if(linksFile!=null){
			inf.network=new LinksHandler();
			inf.network.read(linksFile);
			
			inf.network.printInfo();
		}
		
		inf.build(p,nCommunities,nMaxIteration,outputFile);
	}

	private static void printUsage() {
		System.out
				.println("-a <actionlog> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities>  -l <networkFile> ");
	}
	
}
