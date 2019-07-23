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
import java.util.HashSet;

import util.ArrayUtilities;
import util.IOManager;
import util.Randoms;
import utils.Weka_Utils;
import beans.Activation;
import evaluation.ClusteringEvaluator;
import evaluation.CommunityStructureCalculator;
import evaluation.EvaluateMutualInformation;
import evaluation.Evaluation_Utils;

public class CommunityIC_Inference_Annihilation {
	
	private ActivationsHandler actionLog;
	private int nVertices;
	private Integer[] vertexSet;
	private Int2IntOpenHashMap traceId2Index;
	private Int2IntOpenHashMap vertexId2Index;
	private int nTraces;
	private Integer[] traceSet;
	
	private LinksHandler network;

	static double min_value=0.00000001;
	static double threshold=0.00000001;

	private int save_step=10;

	int nCommunities=-1;
	int nMaxIterations=100;
	int nBurnInIteration=3;
	int nCommunitiesGround;
	
	CommunityIC_Model temp;
	
	double gamma[][];
	double priors[];
	double influenceProbabilities[][];

	private HashMap<Integer, Integer> community_assignments_ground;
	private ArrayList<Integer>[]communities_ground;

	private HashMap<Integer, Integer> assignments;
	private ArrayList<Integer>[]communities;
		 
	
	Randoms rand=new Randoms(13);

	private ArrayList<Activation>[] traces;
	
	static double inf_prob_internal_min=0.01;
	static double inf_prob_internal_max=0.2;

	static double inf_prob_external_max=0.001;
	static double inf_prob_external_min=0.02;

	private HashMap<Integer, Double[]> cache_activation_prob;
	int[]n_adoption_users;
	
	private String outputFile;
	private double N_k;
	
	public void build(ActivationsHandler a,int k,int nMaxIt, String outputFile) throws Exception{
		
		System.out.println("Community IC Inference with "+k+" Communities");
		
		this.actionLog = a;
		this.nMaxIterations=nMaxIt;
		this.nCommunities=k;
		this.outputFile=outputFile;

		
		this.nVertices = a.getNUsers();
		this.vertexSet = a.getUserSet().toArray(new Integer[nVertices]);
		this.vertexId2Index = a.getUserIndex();
		this.traceId2Index = a.getItemIndex();
		this.nTraces = a.getNItems();
		this.traceSet = a.getItemSet().toArray(new Integer[nTraces]);
		this.assignments=new HashMap<Integer, Integer>();
		this.communities=new ArrayList[nCommunities];
		
		this.traces = new ArrayList[nTraces];
		for (int i = 0; i < nTraces; i++) {
			traces[i] = a.getActionsForItem(traceSet[i]);
			Collections.sort(traces[i]);
		}
		this.N_k=Math.sqrt(nVertices);
		//this.N_k=(nVertices);

		init();
		iterate();		
	}//build
	
	
	private void init(){
		System.out.println("Init model...");
		this.priors = new double[nVertices];
		Arrays.fill(priors, min_value);
		
		n_adoption_users=new int[nVertices];
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				n_adoption_users[u]++;
				}
		}
		
		this.influenceProbabilities=new double[nVertices][nCommunities];
		this.gamma=new double[nVertices][nCommunities];
		
		for (int u = 0; u < nVertices; u++) {
			for (int k = 0; k < nCommunities; k++) {
				gamma[u][k] = Math.max(min_value, rand.nextDouble());	
			}
			Weka_Utils.normalize(gamma[u]);
			
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

							
		M_Step(0);
		
		//checkAnnihilation();
		
		System.out.println("------------------------------------------------");
		
		updateAssigments();
		
		if(community_assignments_ground!=null){
			ClusteringEvaluator.evaluate(nCommunitiesGround,nCommunities, community_assignments_ground, assignments);
			EvaluateMutualInformation.evaluate(communities, communities_ground);
		}
		
		if(network!=null){
			CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
			csc.evaluateAll();
		}
		
		
	}//init 
	
	

	
	
	private void checkAnnihilation(){
		int communitiesCounter=0;
		
		for(int c=0;c<priors.length;c++)
			if(priors[c]>0)
				communitiesCounter++;
		
		if(communitiesCounter==nCommunities){
			//do nothing
		}
		else{
			int nAnnihilation=nCommunities-communitiesCounter;
			
			//ricompatto le strutture dati
			double [][]new_gamma=new double[nVertices][communitiesCounter];
			double [][]new_influence_probs=new double[nVertices][communitiesCounter];
			
			int index=0; //indice per la copia
			double priors_new[]=new double[communitiesCounter];
			for(int c=0;c<priors.length;c++){
				if(priors[c]>0){
					//copy 
					for(int u=0;u<nVertices;u++){
						new_gamma[u][index]=gamma[u][c];
						new_influence_probs[u][index]=influenceProbabilities[u][c];
					}
					priors_new[index]=priors[c];
					index++;
				}
			}
			
			priors=priors_new;
			// normalizzo le priors.	
		
			Weka_Utils.normalize(priors);
			
			//aggiorno le strutture. 
			gamma=new_gamma;
			influenceProbabilities=new_influence_probs;
			//aggiorno il numero di comunita'
			nCommunities=communitiesCounter;
			
			System.out.println("Annihilation of "+nAnnihilation+" communities. Now we have "+nCommunities+" communities");
		}
	}//checkAnnihilation
	
	
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
		}

		return countSwitch;
				
	}//updateAssigments
	
	private void iterate(){
		
		System.out.println("Learning phase:\tstarting");
		System.out.println("#Iteration\tChanges\tTime");
	
		double llkold, logLikelihood = 0;
		int saveIteration = 0;
		long initTime = System.currentTimeMillis();
	
	//	logLikelihood = computeLogLikelihood();
	//	System.out.println("Init loglikelihood\t"+logLikelihood);
		
		for (int iterationsDone = 1; iterationsDone <= nMaxIterations; iterationsDone++) {
			llkold = logLikelihood;
			
			E_Step();
			
			int changes=updateAssigments();

			//logLikelihood = computeLogLikelihood();
			//System.out.println("LogLikilihood\t"+logLikelihood);
			
			
			M_Step(iterationsDone);
			
			if(iterationsDone>nBurnInIteration)
				checkAnnihilation();
			
			
			System.out.println(iterationsDone + "\t"+changes+"\t"+(System.currentTimeMillis() - initTime));

			/*
			// else
			if (iterationsDone > 3 && (logLikelihood - llkold) < threshold) {
				System.out.println("Difference Likelihood \t "+(logLikelihood - llkold));
				System.out.println("Stop: Convergence");
		//		break;
			}

			if (iterationsDone > 0 && logLikelihood < llkold) {
				System.out.println("Log likelihood decreases");
				//throw new RuntimeException("LogLikelihood decreases!!!!");
			}*/
			
			saveIteration++;
			if (saveIteration == save_step) {
				saveIteration = 0;
				
				/*temp = buildModel();
				try {
					IOManager.writeObjectOnFile(temp, outputFile + "TMP");
				} catch (Exception e) {
					e.printStackTrace();
				}*/
				
				if(community_assignments_ground!=null){
					ClusteringEvaluator.evaluate(nCommunitiesGround,nCommunities, community_assignments_ground, assignments);
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
	
	
	private double computeLogLikelihood(){
		double logLikelihood=0;
		
		double first_term=0.0;
		double second_term=0.0;
		double third_term=0.0;

		ArrayList<Activation> prev=new ArrayList<Activation>();
		HashSet<Integer> inactive_vertices;
	
		double log_pi[]=new double[nCommunities];
		
		for(int k=0;k<nCommunities;k++)
			if(priors[k]!=0.0)
				log_pi[k]=Math.log(priors[k]);
			else
				throw new RuntimeException("Prior is zero");
		
		
		for(int u=0;u<nVertices;u++)
			for(int k=0;k<nCommunities;k++)
				first_term+=gamma[u][k]*log_pi[k];
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			prev.clear();
			inactive_vertices=new HashSet<Integer>(vertexId2Index.keySet());
			for(Activation a: traces[traceIndex]){
			
				int u=vertexId2Index.get(a.userId);
				if(prev.size()>0){
				
					double eta[][]=computeEta(a, prev);
	
					for(int v_pos=0;v_pos<prev.size();v_pos++){
						Activation v_activation=prev.get(v_pos);
						int v=vertexId2Index.get(v_activation.userId);	
						for(int k=0;k<nCommunities;k++){
							if(influenceProbabilities[v][k]>0||influenceProbabilities[v][k]<1)
								second_term+=eta[v_pos][k]*gamma[u][k]*Math.log(influenceProbabilities[v][k]) - (1-eta[v_pos][k])*Math.log(1-influenceProbabilities[v][k]);
							else{
								ArrayUtilities.print(influenceProbabilities[v]);
								throw new RuntimeException("Influence Prob Error");
								}
						}//for each community
					}//for each influecne v_pos
				}
				prev.add(a);
				inactive_vertices.remove(a.userId);
			}//for each activation on the trace
			
			
			
			for(Activation a:traces[traceIndex]){
				int v=vertexId2Index.get(a.userId);
				double log_failure_influence[]=new double[nCommunities];
				for(int k=0;k<nCommunities;k++)
					log_failure_influence[k]=Math.log(1-influenceProbabilities[v][k]);
				
				for(int inactiveVertexId:inactive_vertices){
					int u=vertexId2Index.get(inactiveVertexId);
					for(int k=0;k<nCommunities;k++)
						if(influenceProbabilities[v][k]<1)
						third_term+=gamma[u][k]*log_failure_influence[k];
						else{
							ArrayUtilities.print(influenceProbabilities[v]);
							throw new RuntimeException("Influence Prob Error");
						}
				}//for each inactive node
				
			}//for each activation
			
		}//for each trace
	
		logLikelihood=first_term+second_term+third_term;
		
		double n=nVertices;
		double K=nCommunities;
		
		
		double sum_k=0;
		for(int k=0;k<nCommunities;k++)
			sum_k+=Math.log(n*priors[k]/12);
		
		double bic_penalization=K/2*Math.log(n/12)+  K*(N_k+1)/2 +  N_k/2*sum_k;
		
		return bic_penalization-logLikelihood;
	}//computeLogLikelihood
	

	
	private double[][]computeEta(Activation a, ArrayList<Activation>influencers){
		double [][]eta=new double[influencers.size()][nCommunities];
		
		double eta_den[]=new double[nCommunities];
		Arrays.fill(eta_den, 1.0);
		
		int v=0;
		for(Activation a_i:influencers){
			int influencerIndex=vertexId2Index.get(a_i.userId);
			for(int k=0;k<nCommunities;k++){
				eta[v][k]+=influenceProbabilities[influencerIndex][k];
				eta_den[k]*=(1-influenceProbabilities[influencerIndex][k]);
			}
			v++;
		}//for each potential influencers
		
		
		for(v=0;v<eta.length;v++)
			for(int k=0;k<nCommunities;k++)
				if(eta_den[k]<1){
					eta[v][k]/=(1-eta_den[k]);
				}
				else{
					ArrayUtilities.print(eta_den);
					throw new RuntimeException("Eta den Exception");
				}
		return eta;
	}//
	
	
	
	private void E_Step(){
		double log_gamma_d_ui[][]=new double[nVertices][nCommunities];
		double log_gamma_not_d_ui[][]=new double[nVertices][nCommunities];

		double log_pi[]=new double[nCommunities];
		for(int k=0;k<nCommunities;k++){
				log_pi[k]=Math.log(priors[k]);
		}
		
		ArrayList<Activation> prev=new ArrayList<Activation>();
		HashSet<Integer> inactive_vertices;

		Double activationprob[]=new Double[nCommunities];

		int activation_index=0;
		cache_activation_prob.clear();
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			prev.clear();
			inactive_vertices=new HashSet<Integer>(vertexId2Index.keySet());
			
			Arrays.fill(activationprob, 1.0);
			double sum_log_failure[]=new double[nCommunities];
			
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				if(prev.size()>0){
				
					for(int k=0;k<nCommunities;k++){
						if(activationprob[k]<0)
							throw new RuntimeException();
						log_gamma_d_ui[u][k]+=Math.log(1-activationprob[k]);
					}
					
				}//prev not empty
				
				prev.add(a);
				inactive_vertices.remove(a.userId);
				
				for(int k=0;k<nCommunities;k++){
					activationprob[k]*=(1-influenceProbabilities[u][k]);
					sum_log_failure[k]+=Math.log(1-influenceProbabilities[u][k]);
				}
				
				Double activation_probs_a[]=new Double[nCommunities];
				System.arraycopy(activationprob, 0, activation_probs_a, 0, activationprob.length);
				
				if(cache_activation_prob.containsKey(activation_index)){
					throw new RuntimeException("Duplicate fhash");
				}
				
				cache_activation_prob.put(activation_index,activation_probs_a);
				
				activation_index++;
				
			}//for each activation on the trace
			
			for(int inactiveVertexId:inactive_vertices){
				int u=vertexId2Index.get(inactiveVertexId);
				for(int k=0;k<nCommunities;k++){
					log_gamma_not_d_ui[u][k]+=sum_log_failure[k];
				}
			}//for each inactive node
	
			
		}//for each trace
		

		for(int u=0;u<nVertices;u++){
			double []log_gammma_u=new double[nCommunities];
			for(int k=0;k<nCommunities;k++)
				log_gammma_u[k]+=log_gamma_d_ui[u][k]+log_gamma_not_d_ui[u][k]+log_pi[k];
				gamma[u]=Weka_Utils.logs2probs(log_gammma_u);	
		}
		
	}//E_Step
	
	private void M_Step(int iterationsDone){
	
		double priors_new[]=new double[nCommunities];
		double Gamma[]=new double[nCommunities];
		double Gamma_i[][]=new double[nTraces][nCommunities];
		
		for(int k=0;k<nCommunities;k++){
			for(int v=0;v<nVertices;v++){
				priors_new[k]+=gamma[v][k];
				Gamma[k]+=gamma[v][k];
			}
			if(iterationsDone>nBurnInIteration)
				priors_new[k]-=N_k/2 ;
		}
		
		Weka_Utils.normalize(priors_new);
		priors=priors_new;
		
		
		double influence_prob_new_num[][]=new double[nVertices][nCommunities];
		double S[][][]=new double[nVertices][nCommunities][2];
		
		
		ArrayList<Activation> prev=new ArrayList<Activation>();
		HashSet<Integer> inactive_vertices;
		
		int activation_index=0;
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			prev.clear();
			inactive_vertices=new HashSet<Integer>(vertexId2Index.keySet());
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				for(int k=0;k<nCommunities;k++)
					Gamma_i[traceIndex][k]+=gamma[u][k];
				
				if(prev.size()>0){
				
					double eta[][];
					
					if(iterationsDone==0){
						eta=computeEta(a, prev);
						
						for(int v_pos=0;v_pos<prev.size();v_pos++){
							Activation v_activation=prev.get(v_pos);
							int v=vertexId2Index.get(v_activation.userId);	
							for(int k=0;k<nCommunities;k++){
								influence_prob_new_num[v][k]+=gamma[u][k]*eta[v_pos][k];
								S[v][k][0]+=gamma[u][k];
							}
						}
					}
					else{
						Double[]activation_prob_ui=cache_activation_prob.get(activation_index);
						if(activation_prob_ui==null)
							throw new RuntimeException("Activation prob cache miss for\t"+a);
						
						for(int v_pos=0;v_pos<prev.size();v_pos++){
							Activation v_activation=prev.get(v_pos);
							int v=vertexId2Index.get(v_activation.userId);	
							for(int k=0;k<nCommunities;k++){
								double eta_v_k=influenceProbabilities[v][k]/(1-activation_prob_ui[k]);
								
								if(eta_v_k<0)
									throw new RuntimeException("eta_v_k cannot be <0");
								
								influence_prob_new_num[v][k]+=gamma[u][k]*eta_v_k;
								S[v][k][0]+=gamma[u][k];
							}
						}// increase the influence prob of influencers. 
					}
					
					
					
				}// prev not empty
				
				prev.add(a);
				inactive_vertices.remove(a.userId);
				activation_index++;
			}//for each activation on the trace
			
			
/*			for(Activation a:traces[traceIndex]){

				int v=vertexId2Index.get(a.userId);
				
				for(int inactiveVertexId:inactive_vertices){
					int u=vertexId2Index.get(inactiveVertexId);
					for(int k=0;k<nCommunities;k++){
						S[v][k][1]+=gamma[u][k];
					}
				}//for each inactive node
				
			}//for each activation	
*/			
		}//for each trace
		
		
		
		for(int u=0;u<nVertices;u++)	
			for(int k=0;k<nCommunities;k++)
				S[u][k][1]=(double)n_adoption_users[u]*Gamma[k];	
						
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				for(int k=0;k<nCommunities;k++)
					S[u][k][1]-=Gamma_i[traceIndex][k];	
			}
		}
		
		for(int u=0;u<nVertices;u++){
			for(int k=0;k<nCommunities;k++){
				double inf_prob_u_k=min_value;
				
				if(influence_prob_new_num[u][k]!=0){
					inf_prob_u_k=influence_prob_new_num[u][k]/(S[u][k][0]+S[u][k][1]);
				}
				if(inf_prob_u_k<min_value)
					inf_prob_u_k=min_value;
				if(inf_prob_u_k>1.0)
					throw new RuntimeException("Inf prob \t"+inf_prob_u_k+" "+influence_prob_new_num[u][k]+" "+S[u][k][0]+" "+S[u][k][1]);
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
		System.out.println(CommunityIC_Inference_Annihilation.class.getSimpleName()
				+ " is running on " + System.getProperty("os.name") + " OS ("
				+ System.getProperty("os.arch") + ")");
		
		
	/*	String network="s3";
		String propagationModel="LT";
		int iterations=100;
		String fold="fold1";
		
		String folder = "resources/datasets/synthetic/"+network+"/"+propagationModel+"/"+fold+"/";
		args = new String[] { "-a", folder + "actionLog", "-o", folder + "C-IC_Model_", "-k", "20",
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

		
		
		CommunityIC_Inference_Annihilation inf=new CommunityIC_Inference_Annihilation();

		if(groundTruthFile!=null){
			inf.community_assignments_ground=Evaluation_Utils.readAssignments(groundTruthFile);
			inf.communities_ground=Evaluation_Utils.readCommunities(groundTruthFile);
			inf.nCommunitiesGround=inf.communities_ground.length;

		}
		
		if(linksFile!=null){
			inf.network=new LinksHandler();
			inf.network.read(linksFile);
			inf.network.printInfo();
		}
		
		inf.build(p,nCommunities,nMaxIteration,outputFile);
	}

	private static void printUsage() {
		System.out.println("-a <actionlog> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities> -l <networkFile> ");
	}
	
}
