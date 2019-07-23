package diffusionBased;

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

import util.ArrayUtilities;
import util.IOManager;
import util.Randoms;
import utils.Weka_Utils;
import beans.Activation;
import evaluation.ClusteringEvaluator;
import evaluation.CommunityStructureCalculator;
import evaluation.EvaluateMutualInformation;
import evaluation.Evaluation_Utils;

public class CommunityRate_Inference_Annihilation {
	
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
	static double min_alpha=0.00000001;

	private int save_step=10;

	int nEffectiveCommunities=-1;
	int nMaxCommunities=-1;
	int nMaxIterations=100;
	int nBurnInIteration=3;
	
	CommunityRateModel temp;
	
	double gamma[][];
	double priors[];
	double alpha[][];

	private HashMap<Integer, Integer> community_assignments_ground;
	private int nEffectiveCommunitiesGround=0;
	private ArrayList<Integer>[]communities_ground;

	private HashMap<Integer, Integer> assignments;
	private ArrayList<Integer>[]communities;
		 
	
	Randoms rand=new Randoms(13);

	private ArrayList<Activation>[] traces;
	
	static double alpha_internal_min=2;
	static double alpha_internal_max=5;

	static double alpha_external_max=0.1;
	static double alpha_external_min=0.5;

	
	private long T=Long.MIN_VALUE;
	static long T_increment=1;
	
	private String outputFile;
	private double avgActivationTime;
	
	
	int[]n_adoption_users;

	
	private HashMap<Integer, Double[]> A_i_u;
	
	
	public void build(ActivationsHandler a,int max_k,int nMaxIt, String outputFile) throws Exception{
		
		System.out.println("Community Net Rate Inference with max "+max_k+" Communities");
		
		this.actionLog = a;
		this.nMaxIterations=nMaxIt;
		this.nMaxCommunities=max_k;
		this.nEffectiveCommunities=nMaxCommunities;
		this.outputFile=outputFile;

		this.nVertices = a.getNUsers();
		this.vertexSet = a.getUserSet().toArray(new Integer[nVertices]);
		this.vertexId2Index = a.getUserIndex();
		this.traceId2Index = a.getItemIndex();
		this.nTraces = a.getNItems();
		this.traceSet = a.getItemSet().toArray(new Integer[nTraces]);
		this.assignments=new HashMap<Integer, Integer>();
		this.communities=new ArrayList[nEffectiveCommunities];
		
		this.traces = new ArrayList[nTraces];
		for (int i = 0; i < nTraces; i++) {
			traces[i] = a.getActionsForItem(traceSet[i]);
			Collections.sort(traces[i]);
			Activation last=traces[i].get(traces[i].size()-1);
			if(last.timeStamp>T)
				T=last.timeStamp;
		}
		
		
		T+=T_increment;
		this.avgActivationTime=(double)T/a.getSize();
		System.out.println("Average activation time\t"+avgActivationTime);

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
		
		this.alpha=new double[nVertices][nEffectiveCommunities];
		this.gamma=new double[nVertices][nEffectiveCommunities];
		
		for (int u = 0; u < nVertices; u++) {
			for (int k = 0; k < nEffectiveCommunities; k++) {
				gamma[u][k] = Math.max(min_value, rand.nextDouble());	
			}
			Weka_Utils.normalize(gamma[u]);
		}
		
		for (int u = 0; u < nVertices; u++){
			int z_u=Weka_Utils.maxIndex(gamma[u]);
			for (int c = 0; c < nEffectiveCommunities; c++)
				if(z_u==c){
					alpha[u][c]=rand.nextUniform(alpha_internal_min, alpha_internal_max);
				}
				else{
					alpha[u][c]=rand.nextUniform(alpha_external_min, alpha_external_max);
				}
		}
		
		
		//fill the cache
			A_i_u=new HashMap<Integer, Double[]>();
			
			int activation_index=0;
			for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
				
				Double cum_a[]=new Double[nEffectiveCommunities];
				Arrays.fill(cum_a, 0.0);
				
				for(Activation a: traces[traceIndex]){
					
					int u=vertexId2Index.get(a.userId);
					
					Double a_i_u[]=new Double[nEffectiveCommunities];
					System.arraycopy(cum_a, 0, a_i_u, 0, cum_a.length);
					
					if(A_i_u.containsKey(activation_index)){
						throw new RuntimeException("Duplicate hash");
					}
					
					A_i_u.put(activation_index,a_i_u);
					
					
					for(int k=0;k<nEffectiveCommunities;k++){
						cum_a[k]+=alpha[u][k];
					}
					
					
					activation_index++;
					
				}//for each activation on the trace
					
			}//for each trace
			
				
		
		
		M_Step(0);
			
		
		System.out.println("------------------------------------------------");
		
		updateAssigments();
		
		if(community_assignments_ground!=null){
			ClusteringEvaluator.evaluate(nEffectiveCommunitiesGround,nEffectiveCommunities, community_assignments_ground, assignments);
			EvaluateMutualInformation.evaluate(communities, communities_ground);
		}
		
		if(network!=null){
			CommunityStructureCalculator csc=new CommunityStructureCalculator(network,assignments,communities);
			csc.evaluateAll();
		}
		
		
	}//init 
	
	
	private int updateAssigments(){
		communities=new ArrayList[nEffectiveCommunities];
		for(int c=0;c<nEffectiveCommunities;c++)
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
		for(int c=0;c<nEffectiveCommunities;c++)
			if(communities[c].size()!=0)
				n_not_empty_Communities++;
		
		if(n_not_empty_Communities<nEffectiveCommunities){
			ArrayList<Integer>comm2[]=new ArrayList[n_not_empty_Communities];
			int counter=0;
			for(int c=0;c<nEffectiveCommunities;c++)
				if(communities[c].size()!=0){
					comm2[counter]=communities[c];
					counter++;
				}
			communities=comm2;
			
			assignments.clear();
			//nEffectiveCommunities=n_not_empty_Communities;
			
			for(int c=0;c<communities.length;c++)
				for(int userId:communities[c])
					assignments.put(userId,c+1);	
		}
		
		
		
		return countSwitch;
		
		
		
	}//updateAssigments
	
	private void iterate(){
		
		
		System.out.println("Learning phase:\tstarting");
		System.out.println("#Iteration \t Changes \t Time");
	
		double llkold, logLikelihood = 0;
		int saveIteration = 0;
		long initTime = System.currentTimeMillis();
	
	//	logLikelihood = computeLogLikelihood();
	//	System.out.println("Init loglikelihood\t"+logLikelihood);
		for (int iterationsDone = 1; iterationsDone <= nMaxIterations; iterationsDone++) {
			llkold = logLikelihood;
			

			
			E_Step();
			int changes=updateAssigments();
			
			
		/*	logLikelihood = computeLogLikelihood();
			System.out.println("LogLikilihood\t"+logLikelihood);*/
			
			
			M_Step(iterationsDone);
			
			
			if(iterationsDone>nBurnInIteration)
				checkAnnihilation();
						
			
			System.out.println(iterationsDone + "\t"+changes+"\t"+(System.currentTimeMillis() - initTime));
			
			
		
			
		/*	// else
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
				//logLikelihood = computeLogLikelihood();
				//System.out.println("Likelihood\t"+logLikelihood);
				
				saveIteration = 0;
				temp = buildModel();
				try {
					IOManager.writeObjectOnFile(temp, outputFile + "TMP");
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				if(community_assignments_ground!=null){
					ClusteringEvaluator.evaluate(nEffectiveCommunitiesGround,nEffectiveCommunities, community_assignments_ground, assignments);
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
			ClusteringEvaluator.evaluate(nEffectiveCommunitiesGround,nEffectiveCommunities, community_assignments_ground, assignments);
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
	
	/*
	private double computeLogLikelihood(){
		double logLikelihood=0;
		
		double first_term=0.0;
		double second_term=0.0;
		double third_term=0.0;
		double fourth_term=0.0;

		ArrayList<Activation> prev=new ArrayList<Activation>();
		HashSet<Integer> inactive_vertices;
	
		double log_pi[]=new double[nEffectiveCommunities];
		
		for(int k=0;k<nEffectiveCommunities;k++)
			if(priors[k]!=0.0)
				log_pi[k]=Math.log(priors[k]);
			else
				throw new RuntimeException("Prior is zero");
		
		
		for(int u=0;u<nVertices;u++)
			for(int k=0;k<nEffectiveCommunities;k++)
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
					
						long delta_u_v=a.timeStamp-v_activation.timeStamp;
						int v=vertexId2Index.get(v_activation.userId);	
						for(int k=0;k<nEffectiveCommunities;k++){
							if(alpha[v][k]>0)
								third_term+=eta[v_pos][k]*gamma[u][k]*Math.log(alpha[v][k]);
							else
								throw new RuntimeException("Alpha zero");
						
							fourth_term+=gamma[u][k]*delta_u_v * alpha[v][k];
						}
						
					}
				}
				prev.add(a);
				inactive_vertices.remove(a.userId);
			}//for each activation on the trace
			
			
			
			for(Activation a:traces[traceIndex]){
				
				long delta_v=T-a.timeStamp;
				int v=vertexId2Index.get(a.userId);
				
				for(int inactiveVertexId:inactive_vertices){
					int u=vertexId2Index.get(inactiveVertexId);
					for(int k=0;k<nEffectiveCommunities;k++)
						second_term+=gamma[u][k]*delta_v*alpha[v][k];
				}//for each inactive node
				
			}//for each activation
			
		}//for each trace
		
		
		double N_k=Math.sqrt(nVertices);
		double n=nVertices;
		double K=nEffectiveCommunities;
		
		
		double sum_k=0;
		for(int k=0;k<nEffectiveCommunities;k++)
			sum_k+=Math.log(n*priors[k]/12);
		
		
		double bic_penalization=K/2*Math.log(n/12)+  K*(N_k+1)/2 +  N_k/2*sum_k;
		
		
		logLikelihood=first_term-second_term+third_term-fourth_term;
		
		return bic_penalization-logLikelihood;
	}//computeLogLikelihood
	

	
	private double[][]computeEta(Activation a, ArrayList<Activation>influencers){
		double [][]eta=new double[influencers.size()][nEffectiveCommunities];
		
		double eta_den[]=new double[nEffectiveCommunities];
		int v=0;
		for(Activation a_i:influencers){
			double hazards_v[]=computeHazard(a, a_i);
			for(int k=0;k<nEffectiveCommunities;k++){
				eta[v][k]+=hazards_v[k];
				eta_den[k]+=hazards_v[k];
			}
			v++;
		}//for each potential influencers
		
		
		for(v=0;v<eta.length;v++)
			for(int k=0;k<nEffectiveCommunities;k++)
				if(eta_den[k]!=0.0)
					eta[v][k]/=eta_den[k];
				else
					throw new RuntimeException("Eta den is zero");
		return eta;
	}//
	
	
	private double[][]initEta(Activation a, ArrayList<Activation>influencers){
		double [][]eta=new double[influencers.size()][nEffectiveCommunities];
		for(int k=0;k<nEffectiveCommunities;k++){
			int index_influencer_in_k=-1;
			int cnt=0;
			for(Activation a_i:influencers){
				int v=vertexId2Index.get(a_i.userId);
				int z_v=Weka_Utils.maxIndex(gamma[v]);
				if(z_v==k)
					index_influencer_in_k=cnt;
				cnt++;
			}
			
			if(index_influencer_in_k>0)
				eta[index_influencer_in_k][k]=1.0;
			else{
				//random
				double sum=0.0;
				for(int v=0;v<eta.length;v++){
					eta[v][k]=rand.nextDouble();
					sum+=eta[v][k];
				}
				for(int v=0;v<eta.length;v++)
					eta[v][k]/=sum;
				
			}
		}
	
		
		return eta;
	}//
	
	
	
	
	private double[] computeHazard(Activation a, Activation prev){
		int v=vertexId2Index.get(prev.userId);
		return alpha[v];
	}//compute hazard
	
	
	private double[] computeLogSurvival(Activation a, Activation prev){
		double []ris=new double[nEffectiveCommunities];
		
		long delta_t=a.timeStamp-prev.timeStamp;
		
		int v=vertexId2Index.get(prev.userId);
		
		for(int k=0;k<nEffectiveCommunities;k++)
			ris[k]=-alpha[v][k]*delta_t;
		
		return ris;
	}
	
	
	private double[] computeLogSurvival(Activation prev){
		double []ris=new double[nEffectiveCommunities];
		
		long delta_t=T-prev.timeStamp;
		
		int v=vertexId2Index.get(prev.userId);
		
		for(int k=0;k<nEffectiveCommunities;k++)
			ris[k]=-alpha[v][k]*delta_t;
		
		return ris;
	}
	
	
	
	private double [] computeSurvival(Activation a, Activation prev){
		double ris[]=computeLogSurvival(a, prev);
		for(int k=0;k<nEffectiveCommunities;k++)
			ris[k]=Math.exp(ris[k]);
		return ris;
	}
	
	*/
	
	private void E_Step(){
		
		double log_pi[]=new double[nEffectiveCommunities];
		for(int k=0;k<nEffectiveCommunities;k++)
			log_pi[k]=Math.log(priors[k]);
		
		A_i_u.clear();
		
		double A_k[]=new double[nEffectiveCommunities];
		double A_u_k[][]=new double[nVertices][nEffectiveCommunities];
		double A_i_k[][]=new double[nTraces][nEffectiveCommunities];
		double A_tilde_u_k[][]=new double[nVertices][nEffectiveCommunities];
		double A_log_tilde_u_k[][]=new double[nVertices][nEffectiveCommunities];

		double B_i_k[][]=new double[nTraces][nEffectiveCommunities];
		double B_k[]=new double[nEffectiveCommunities];
		double B_u_k[][]=new double[nVertices][nEffectiveCommunities];
		double B_tilde_u_k[][]=new double[nVertices][nEffectiveCommunities];

		
		int activation_index=0;
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			
			Double cum_a[]=new Double[nEffectiveCommunities];
			Double cum_b[]=new Double[nEffectiveCommunities];

			Arrays.fill(cum_a, 0.0);
			Arrays.fill(cum_b, 0.0);

			int cnt_trace=0;
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				Double a_i_u[]=new Double[nEffectiveCommunities];
				System.arraycopy(cum_a, 0, a_i_u, 0, cum_a.length);
				
				if(A_i_u.containsKey(activation_index)){
					throw new RuntimeException("Duplicate hash");
				}
				A_i_u.put(activation_index,a_i_u);
				
				for(int k=0;k<nEffectiveCommunities;k++){
					A_i_k[traceIndex][k]+=alpha[u][k];
					A_k[k]+=alpha[u][k];
					B_i_k[traceIndex][k]+=alpha[u][k]*a.timeStamp;
					B_k[k]+=alpha[u][k]*a.timeStamp;
					
					A_tilde_u_k[u][k]+=a_i_u[k]*a.timeStamp;
					
					if(cnt_trace>0)
						A_log_tilde_u_k[u][k]+=Math.log(a_i_u[k]);
					
					B_tilde_u_k[u][k]+=cum_b[k];
					
					
					cum_a[k]+=alpha[u][k];
					cum_b[k]+=alpha[u][k]*a.timeStamp;
				}
				
				activation_index++;
				cnt_trace++;
			}//for each activation on the trace
				
		}//for each trace
		
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				for(int k=0;k<nEffectiveCommunities;k++){
					A_u_k[u][k]+=A_i_k[traceIndex][k];
					B_u_k[u][k]+=B_i_k[traceIndex][k];
				}
								
			}//for each activation on the trace
				
		}//for each trace
		
		
		for(int u=0;u<nVertices;u++){
			double []log_gammma_u=new double[nEffectiveCommunities];
			for(int k=0;k<nEffectiveCommunities;k++){
				log_gammma_u[k]= B_k[k]
								-B_u_k[u][k]
								-T*(A_k[k]-A_u_k[u][k])
								+B_tilde_u_k[u][k]
								-A_tilde_u_k[u][k]
								+A_log_tilde_u_k[u][k] +log_pi[k];
			}
			try{
				gamma[u]=Weka_Utils.logs2probs(log_gammma_u);	
			}catch(Exception e ){
				System.out.println("Error gamma");
				System.out.println("Log gamma");
				ArrayUtilities.print(log_gammma_u);
				System.out.println("log pi");
				ArrayUtilities.print(log_pi);

				e.printStackTrace();
				System.exit(-1);
			}
		}
	
		
	}//E_Step
	
	private void M_Step(int iterationsDone){
		double priors_new[]=new double[nEffectiveCommunities];
	
		for(int k=0;k<nEffectiveCommunities;k++){
			for(int v=0;v<nVertices;v++)
				priors_new[k]+=gamma[v][k];
			//priors_new[k]-=Math.sqrt(nVertices)/2 * Math.min(1, 1d/(H-iteration));
			if(iterationsDone>nBurnInIteration)
				priors_new[k]-=Math.sqrt(nVertices)/2 ;

		}
		
		Weka_Utils.normalize(priors_new);
		priors=priors_new;
		
		double Gamma[]=new double[nEffectiveCommunities];
		
		for(int u=0;u<nVertices;u++)
			for(int k=0;k<nEffectiveCommunities;k++)
				Gamma[k]+=gamma[u][k];
		
		double Gamma_i_k[][]=new double[nTraces][nEffectiveCommunities];
		double tau_v[]=new double[nVertices];
		double Gamma_v_k[][]=new double[nVertices][nEffectiveCommunities];
		double C_i_k[][]=new double[nTraces][nEffectiveCommunities];
		double C_v_k[][]=new double[nVertices][nEffectiveCommunities];
		double D_v_k[][]=new double[nVertices][nEffectiveCommunities];
		double E_v_k[][]=new double[nVertices][nEffectiveCommunities];
		
		double E_i_k[][]=new double[nTraces][nEffectiveCommunities];

		double F_v_k[][]=new double[nVertices][nEffectiveCommunities];
		double Psi_v_k[][]=new double[nVertices][nEffectiveCommunities];

		
		int adoption_index=0;
		
		//fill C_i_k & D_v_k E_i_k
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			double d_cum_trace[]=new double[nEffectiveCommunities];
			double f_cum_trace[]=new double[nEffectiveCommunities];

			double gamma_cum_trace[]=new double[nEffectiveCommunities];
			int trace_cnt=0;
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				Double a_i_u[]=A_i_u.get(adoption_index);
				if(a_i_u==null)
					throw new RuntimeException("Cache miss for \t"+a);
				
			
				for(int k=0;k<nEffectiveCommunities;k++){
					Gamma_i_k[traceIndex][k]+=gamma[u][k];
					
					
					if(trace_cnt>0){
						C_i_k[traceIndex][k]+=gamma[u][k]/a_i_u[k];
						d_cum_trace[k]+=gamma[u][k]/a_i_u[k];
					}
						
					D_v_k[u][k]+=d_cum_trace[k];
					
					gamma_cum_trace[k]+=gamma[u][k];
					Psi_v_k[u][k]+=a.timeStamp*gamma_cum_trace[k];
					E_i_k[traceIndex][k]+=gamma[u][k]*a.timeStamp;
					
					//the orded matters
					f_cum_trace[k]+=gamma[u][k]*a.timeStamp;
					F_v_k[u][k]+=f_cum_trace[k];
				}
				
				tau_v[u]+=a.timeStamp;
				
				adoption_index++;
				trace_cnt++;
			}//for each activation on the trace
			
		}//first pass
		
		adoption_index=-1;
		//fill C_u_k	
		for(int traceIndex=0;traceIndex<nTraces;traceIndex++){
			for(Activation a: traces[traceIndex]){
				int u=vertexId2Index.get(a.userId);
				
				for(int k=0;k<nEffectiveCommunities;k++){
					Gamma_v_k[u][k]+=Gamma_i_k[traceIndex][k];
					C_v_k[u][k]+=C_i_k[traceIndex][k];
					E_v_k[u][k]+=E_i_k[traceIndex][k];
				}
				
			}//for each activation on the trace
			
		}//second iteration
			

		for(int v=0;v<nVertices;v++){
			for(int k=0;k<nEffectiveCommunities;k++){
				
				double alpha_u_k_num=alpha[v][k]*(C_v_k[v][k]-D_v_k[v][k]);
		
				if(Double.isNaN(alpha_u_k_num)){
					
					System.out.println(alpha[v][k]);
					System.out.println(C_v_k[v][k]);
					System.out.println(D_v_k[v][k]);
					throw new RuntimeException();
					
				}
				
				double alpha_u_k_den=(T*n_adoption_users[v]-tau_v[v])*Gamma[k]
									  -T*Gamma_v_k[v][k]
									   +E_v_k[v][k]-F_v_k[v][k]+Psi_v_k[v][k];
				
				if(alpha_u_k_den!=0.0){
					if(alpha_u_k_num!=0.0)
						alpha[v][k]=alpha_u_k_num/alpha_u_k_den;
				}
				
				if(Double.isNaN(alpha[v][k])){
					System.out.println("Error update alpha");
					System.out.println("alpha_u_k_num\t"+alpha_u_k_num);
					System.out.println("alpha_u_k_den\t"+alpha_u_k_den);
					throw new RuntimeException();
				}
				if(alpha[v][k]<min_alpha)
					alpha[v][k]=min_alpha;
				}
			
				
			}	
		
	}//M-step
	
	
	
	private void checkAnnihilation(){
		int communitiesCounter=0;
		
		for(int c=0;c<priors.length;c++)
			if(priors[c]>0)
				communitiesCounter++;
		
		if(communitiesCounter==nEffectiveCommunities){
			//do nothing
		}
		else{
			int nAnnihilation=nEffectiveCommunities-communitiesCounter;
			
			//ricompatto le strutture dati
			double [][]new_gamma=new double[nVertices][communitiesCounter];
			double [][]new_alpha=new double[nVertices][communitiesCounter];
			
			int index=0; //indice per la copia
			double priors_new[]=new double[communitiesCounter];
			for(int c=0;c<priors.length;c++){
				if(priors[c]>0){
					//copy 
					for(int u=0;u<nVertices;u++){
						new_gamma[u][index]=gamma[u][c];
						new_alpha[u][index]=alpha[u][c];
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
			alpha=new_alpha;
			//aggiorno il numero di comunita'
			nEffectiveCommunities=communitiesCounter;
			
			System.out.println("Annihilation of "+nAnnihilation+" communities. Now we have "+nEffectiveCommunities+" communities");
		}
	}//checkAnnihilation
	
	
	private CommunityRateModel buildModel(){
		temp=new CommunityRateModel();
		temp.setAlpha(alpha);
		temp.setnTraces(nTraces);
		temp.setnVertices(nVertices);
		temp.setTraceId2Index(traceId2Index);
		temp.setTraceSet(traceSet);
		temp.setVertexId2Index(vertexId2Index);
		temp.setVertexSet(vertexSet);
		temp.setGamma(gamma);
		return temp;
	}//buildModel
	
	
	
	public static void main(String[] args) throws Exception {
		System.out.println("Community Rate Inference with Annihilation");
		
		
	/*	String network="s1";
		String propagationModel="LT";
		int iterations=100;
		String fold="fold1";
		
		String folder = "resources/datasets/synthetic/"+network+"/"+propagationModel+"/"+fold+"/";
		args = new String[] { "-a", folder + "actionLog", "-o", folder + "C-NetRATE_Model_", "-k", "20",
				"-maxIt", ""+iterations, "-g", "resources/datasets/synthetic/"+network+"/community.dat" };
		*/
	/*	String folder="resources/datasets/flixster/";
		int iterations=100;
		args=new String[]{"-a", folder + "actionLog", "-o", folder + "C-NetRATE_Model_", "-k", "50",
				"-maxIt", ""+iterations, "-l", folder+"links"};
		*/
		
		String actionsFile = null;
		int nEffectiveCommunities = 3;
		int nMaxIteration = 20;
		String groundTruthFile=null;
		String confFile=null;
		// long deltaT=1000;
		String outputFile = "C-Rate_Model";
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
				nEffectiveCommunities = Integer.parseInt(args[i + 1]);
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

		
		
		CommunityRate_Inference_Annihilation crate=new CommunityRate_Inference_Annihilation();

		if(groundTruthFile!=null){
			crate.community_assignments_ground=Evaluation_Utils.readAssignments(groundTruthFile);
			crate.communities_ground=Evaluation_Utils.readCommunities(groundTruthFile);
			crate.nEffectiveCommunitiesGround=crate.communities_ground.length;
		}
		
		if(linksFile!=null){
			crate.network=new LinksHandler();
			crate.network.read(linksFile);
			crate.network.printInfo();
		}
		
		crate.build(p,nEffectiveCommunities,nMaxIteration,outputFile);
	}

	private static void printUsage() {
		System.out
				.println("-a <actionlog> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities>  -l <networkFile> ");
	}
	
}
