package topicBased;

import handlers.ActivationsHandler;
import handlers.LinksHandler;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import javax.sound.midi.SysexMessage;

import util.IOManager;
import utils.Weka_Utils;
import beans.Activation;
import evaluation.ClusteringEvaluator;
import evaluation.CommunityStructureCalculator;
import evaluation.EvaluateMutualInformation;
import evaluation.Evaluation_Utils;

public class Bernullian_Inference {

	
	private ActivationsHandler propagationLog;
	private int nVertices;
	private Integer[] vertexSet;
	private Int2IntOpenHashMap traceId2Index;
	private Int2IntOpenHashMap vertexId2Index;
	private int nTraces;
	private Integer[] traceSet;
	
	private LinksHandler network;

	
	private HashMap<Integer, Integer> community_assignments_ground;
	private ArrayList<Integer>[]communities_ground;

	private HashMap<Integer, Integer> assignments;
	private ArrayList<Integer>[]communities;
	
	
	
	static double min_prob_value=0.0001;
	static double max_prob_value=0.95;
	
	static double threshold=0.000001;
	private int save_step=10;

	Multinomial_Model temp;
	
	double pi[];
	
	double gamma[][];
	double phi[][];
	int nCommunities=-1;
	int nMaxIterations=100;
	
	
	Random rand=new Random(System.currentTimeMillis());
	private String outputFile;
	
	public void build(ActivationsHandler a,int k, String outputFile,int nMaxIterations) throws Exception{
		
		this.propagationLog = a;
		this.nVertices = a.getNUsers();
		this.vertexSet = a.getUserSet().toArray(new Integer[nVertices]);
		this.vertexId2Index = a.getUserIndex();
		this.traceId2Index = a.getItemIndex();

		this.nTraces = a.getNItems();
		this.traceSet = a.getItemSet().toArray(new Integer[nTraces]);
		
		this.nCommunities=k;
		this.nMaxIterations=nMaxIterations;
		this.outputFile=outputFile;
		System.out.println("Multinomial Model Inference with "+nCommunities+" Topics");
		
		init();
	
		iterate();
	
		
	}//build
	
	public void storeResults() throws Exception{
		
	}
	
	private void init() {
		System.out.println("Init model...");
		
		gamma=new double[nVertices][nCommunities];
		phi=new double[nCommunities][nTraces];
		
		pi=new double[nCommunities];
	
		this.assignments=new HashMap<Integer, Integer>();
		
		
		for(int v=0;v<nVertices;v++){
			for(int k=0;k<nCommunities;k++)
				gamma[v][k]=rand.nextDouble();
			Weka_Utils.normalize(gamma[v]);
			for(int k=0;k<nCommunities;k++)
				pi[k]+=gamma[v][k];
		}
		
		
		Weka_Utils.normalize(pi);
		
		for(int k=0;k<nCommunities;k++){
			for(int i=0;i<nTraces;i++){
				phi[k][i]=rand.nextDouble();
				if(phi[k][i]<min_prob_value)
					phi[k][i]=min_prob_value;
			}
		}
		
		updateAssigments();
		System.out.println("------------------------------------------------");
		
		
		if(community_assignments_ground!=null){
			ClusteringEvaluator.evaluate(nCommunities, community_assignments_ground, assignments);
			EvaluateMutualInformation.evaluate(communities, communities_ground);
		}
		
		if(network!=null){
			CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
			csc.evaluateAll();
		}
		
		
	}//init
	
	
	private void iterate() {
		System.out.println("Learning phase:\tstarting");
		System.out.println("#Iteration\tChanges\tTime");
	
		

		double llkold, logLikelihood = 0;
		int saveIteration = 0;
		long initTime = System.currentTimeMillis();
	
		for (int iterationsDone = 1; iterationsDone <= nMaxIterations; iterationsDone++) {
			llkold = logLikelihood;

		
			E_Step();
		/*	logLikelihood = computeLogLikelihood();
			System.out.println("Likelihood\t"+logLikelihood);*/
			
			int changes=updateAssigments();
			M_Step();
			
			System.out.println(iterationsDone + "\t"+changes+"\t"+(System.currentTimeMillis() - initTime));

			
			/*// else
			if (iterationsDone > 3 && (logLikelihood - llkold) < threshold) {
				System.out.println("Difference Likelihood \t "+(logLikelihood - llkold));
				System.out.println("Stop: Convergence");
				break;
			}

			if (iterationsDone > 0 && logLikelihood < llkold) {
				throw new RuntimeException("LogLikelihood decreases!!!!");
			}
*/
			saveIteration++;
			if (saveIteration == save_step) {
				saveIteration = 0;
				
			/*	logLikelihood = computeLogLikelihood();
				System.out.println("Likelihood\t"+logLikelihood);
				
				
				temp = buildModel();
				try {
					IOManager.writeObjectOnFile(temp, outputFile + "TMP_"+nCommunities);
				} catch (Exception e) {
					e.printStackTrace();
				}
				*/
				
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
			File f=new File(outputFile + "TMP_"+nCommunities);
			f.delete();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(outputFile
					+ "Communities"));
			pw.println("Vertex\tCommunity");
			for (int u = 0; u < nVertices; u++) {
				pw.println("" + vertexSet[u] + "\t"
						+ (Weka_Utils.maxIndex(gamma[u])+1) );
			}
			pw.flush();
			pw.close();

		} catch (IOException e) {
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
		
		System.out.print("DONE");
		
	}//iterate

	
	
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
	
	
	private Multinomial_Model buildModel() {
		Multinomial_Model mm=new Multinomial_Model();
		mm.setGamma(gamma);
		mm.setK(nCommunities);
		mm.setnTraces(nTraces);
		mm.setnVertices(nVertices);
		mm.setPhi(phi);
		mm.setTraceId2Index(traceId2Index);
		mm.setTraceSet(traceSet);
		mm.setVertexId2Index(vertexId2Index);
		mm.setVertexSet(vertexSet);
		mm.setPi(pi);
		return mm;
	}//buildModel

	
	private double computeLogLikelihood() {
		double logLikelihood=0.0;
		
		double log_pi[]=new double[nCommunities];
	
	
		for(int k=0;k<nCommunities;k++)
			log_pi[k]=Math.log(pi[k]);
		
		ArrayList<Activation> activationForVertex;
		HashSet<Integer> allTraces;
		
		for(int v=0;v<nVertices;v++){
			activationForVertex=propagationLog.getActionsForUser(vertexSet[v]);
			
			allTraces=new HashSet<Integer>(traceId2Index.keySet());
			double llk_v=0.0;
		
			for(int k=0;k<nCommunities;k++)
				llk_v+=gamma[v][k]* log_pi[k];
			
			
			for(Activation a: activationForVertex){
				int traceId=a.itemId;
				allTraces.remove(traceId);
				int i=traceId2Index.get(traceId);
				for(int k=0;k<nCommunities;k++){
					llk_v+=gamma[v][k]*Math.log(phi[k][i]);
				}
			}
			
			
			for(int traceId:allTraces){
				int i=traceId2Index.get(traceId);
				for(int k=0;k<nCommunities;k++){
					llk_v+=gamma[v][k]*Math.log(1-phi[k][i]);
				}
			}
			
			logLikelihood+=llk_v;
		}//for each vertex
		
		return logLikelihood;
	}//computeLogLikelihood
	
	
	private void M_Step() {
	
		double pi_new[]=new double[nCommunities];
		Arrays.fill(pi_new, min_prob_value);
		
		double sum_gamma[]=new double[nCommunities];
		
		
		for(int v=0;v<nVertices;v++){
			for(int k=0;k<nCommunities;k++){
				pi_new[k]+=gamma[v][k];
				sum_gamma[k]+=gamma[v][k];
			}
		}
		
		Weka_Utils.normalize(pi_new);
		
		
		double phi_new[][]=new double[nCommunities][nTraces];
		
		ArrayList<Activation> activations_for_trace=null;
		for(int i=0;i<nTraces;i++){
			activations_for_trace=propagationLog.getActionsForItem(traceSet[i]);
			
			for(Activation a:activations_for_trace){
				int v=vertexId2Index.get(a.userId);
				for(int k=0;k<nCommunities;k++)
					phi_new[k][i]+=gamma[v][k];
				
			}	
		}
		
		for(int i=0;i<nTraces;i++){
			for(int k=0;k<nCommunities;k++){
				if(phi_new[k][i]>min_prob_value){
					if(sum_gamma[k]>0)
						phi_new[k][i]=phi_new[k][i]/sum_gamma[k];
					else
						throw new RuntimeException();
				}
				else{
					phi_new[k][i]=min_prob_value;
				}
				
				if(phi_new[k][i]>=max_prob_value){
					phi_new[k][i]=max_prob_value;
				}
				
				if(phi_new[k][i]<=min_prob_value){
					phi_new[k][i]=min_prob_value;
				}
			}//for each k
		}//for eacht trace
	
	
		phi=phi_new;
		pi=pi_new;
	}//M_Step

	private void E_Step() {
		double log_gamma_new[][]=new double[nVertices][nCommunities];
		
		
		double log_pi[]=new double[nCommunities];
		for(int k=0;k<nCommunities;k++){
			if(pi[k]==0.0)
				throw new RuntimeException();
			log_pi[k]=Math.log(pi[k]);
		}
		
		ArrayList<Activation>activationForVertex;
		HashSet<Integer>allTraces;
		for(int v=0;v<nVertices;v++){
			activationForVertex=propagationLog.getActionsForUser(vertexSet[v]);
			
			allTraces=new HashSet<Integer>(traceId2Index.keySet());
			
		
			for(int k=0;k<nCommunities;k++)
				log_gamma_new[v][k]+=log_pi[k];
			
			
			for(Activation a: activationForVertex){
				int traceId=a.itemId;
				allTraces.remove(traceId);
				int i=traceId2Index.get(traceId);
				for(int k=0;k<nCommunities;k++){
					log_gamma_new[v][k]+=Math.log(phi[k][i]);
					if(Double.isNaN(log_gamma_new[v][k])){
						System.out.println(phi[k][i]);
						throw new RuntimeException();
					}
				}
			}
			
			
			for(int traceId:allTraces){
				int i=traceId2Index.get(traceId);
				for(int k=0;k<nCommunities;k++){
					log_gamma_new[v][k]+=Math.log(1.0-phi[k][i]);
					if(Double.isNaN(log_gamma_new[v][k])){
						System.out.println(1-phi[k][i]);
						throw new RuntimeException();
					}
				}
			}
			
		}//for each v
		
		for(int v=0;v<nVertices;v++)
			gamma[v]=Weka_Utils.logs2probs(log_gamma_new[v]);
		
	}//E-step

	
	public double[][]getGamma(){
		return gamma;
	}
	
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		System.out.println(Bernullian_Inference.class.getSimpleName()
				+ " is running on " + System.getProperty("os.name") + " OS ("
				+ System.getProperty("os.arch") + ")");
		
		/*	String network="s1";
		String propagationModel="NR";
		int iterations=100;
		String fold="fold1";
		
		String folder = "resources/datasets/synthetic/"+network+"/"+propagationModel+"/"+fold+"/";
		args = new String[] { "-a", folder + "actionLog", "-o", folder + "CTM_Model", "-k", "9",
				"-maxIt", ""+iterations, "-g", "resources/datasets/synthetic/"+network+"/community.dat" };
		
		*/
		
		String confFile = null;
		String actionsFile = null;
		String groundTruthFile=null;
		String linksFile=null;
		int topics=0;
		int nMaxIteration = 100;
		
		String outputFile="";
		
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
			if (args[i].equals("-c")) {
				confFile = args[i + 1];
				i++;
			}
			if (args[i].equals("-k")) {
				topics = Integer.parseInt(args[i + 1]);
				i++;
			}
			if (args[i].equals("-o")) {
				outputFile = args[i + 1];
				i++;
			}
			if (args[i].equals("-it")) {
				nMaxIteration = Integer.parseInt(args[i + 1]);
				i++;
			}
			
			if (args[i].equals("-g")) {
				groundTruthFile =args[i + 1];
				i++;
			}

			
			if (args[i].equals("-l")) {
				linksFile =args[i + 1];
				i++;
			}

			
			
		}// for each arg
		
		
		if(outputFile.length()==0){
			outputFile="Mixture_Model_"+topics;
		}
		
		
		ActivationsHandler p = new ActivationsHandler();
		p.read(actionsFile, confFile);
		p.printInfo();
		
		
		
		Bernullian_Inference mi=new Bernullian_Inference();
		if(groundTruthFile!=null){
			mi.community_assignments_ground=Evaluation_Utils.readAssignments(groundTruthFile);
			mi.communities_ground=Evaluation_Utils.readCommunities(groundTruthFile);
		}
		
		if(linksFile!=null){
			mi.network=new LinksHandler();
			mi.network.read(linksFile);
			mi.network.printInfo();
		}
		
		mi.build(p, topics,outputFile,nMaxIteration);
	
	}//main

	
	private static void printUsage() {
		System.out.println("-a <actionlog> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities> -l <networkFile>");
	}
	
	
}//Multinomial_Inference
