package propagationGenerator;

import handlers.CommunityAssignmentHandler;
import handlers.LinksHandler;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;


import beans.Activation;
import beans.Activation_Influence;
import beans.Edge;

public class PropagationGenerator {

	private int nPropagations = 1000;
	private int minLenght = 5;
	private long seed = 11;
	
	private ArrayList<Activation_Influence>[] traces;

	protected LinksHandler network;
	private CommunityAssignmentHandler assignment;
	protected Map<Edge, Double> influenceWeights;
	
	
	PropagationModel pm=new IC_Rate_PropagationModel();
	protected double communityMix=0;

	Random rand=new Random(System.currentTimeMillis());
	
	public void generate(){

		int nNodes=network.getNVertices();
		
		HashSet<Edge>sigmaEdges=new HashSet<Edge>();
		
		ArrayList<Integer>[]communities=assignment.getCommunities();
		for(int k=0;k<communities.length;k++){
			for(int vertex:communities[k]){
				sigmaEdges.add(new Edge(-(k+1),vertex));
			}	
		}//for each community
		network.addEdges(sigmaEdges);
		
		pm.setNetwork(network);
	
		System.out.println("------------------------------------------------");
		System.out.println("Generating propagation...");
		traces=new ArrayList[nPropagations];
		
		HashSet<Integer>activeUsers=new HashSet<Integer>();
		pm.initialize();
		
		int countTrials=0;
		int maxTrials=1000;
	
		ArrayList<Activation_Influence>[] best_traceGeneration = null;
		int best_trace_Generation_score=-1;
		do{
			pm.initialize();
			activeUsers.clear();
			countTrials++;
			
			for (int traceId = 0; traceId < nPropagations; traceId++) {
		
				HashSet<Integer>seedCommunities=new HashSet<Integer>();
				int cnt=0;
				do{
					cnt++;
					ArrayList<Integer>communityList=new ArrayList<Integer>();
					for(int i=0;i<assignment.getnCommunities();i++)
						communityList.add(-1*(i+1));
					Collections.shuffle(communityList);
					
					do{
						seedCommunities.clear();
						double prob=rand.nextDouble();
						for(int community:communityList){
							if(rand.nextDouble()<prob){
								seedCommunities.add(community);
								prob=prob*communityMix;
							}
						}
					}while(seedCommunities.size()==0);
					
					
					traces[traceId]=pm.generateTrace(seedCommunities,traceId);
					Collections.sort(traces[traceId]);
					if(cnt==1000){
						System.out.println("Current trace lenght\t"+traces[traceId].size());
						throw new RuntimeException("Cannot generate trace\t"+traceId);
					}
				
				}while(traces[traceId]==null||traces[traceId].size()<minLenght);
			//System.out.println("Generated trace\t"+traceId);	
			cnt=0;
			HashSet<Integer>users_for_trace=getUsersForTrace(traces[traceId]);
			activeUsers.addAll(users_for_trace);
			}//for each trace to be generate
			int covered_nodes=activeUsers.size();
			
			System.out.println("Covered Nodes\t"+covered_nodes);
			
			double avg_lenght=0.0;
			for(int i=0;i<nPropagations;i++)
				avg_lenght+=traces[i].size();
			avg_lenght/=nPropagations;
			//	System.out.println("Average length of the traces\t"+avg_lenght);
			
			if(covered_nodes>best_trace_Generation_score){
				best_trace_Generation_score=covered_nodes;
				best_traceGeneration=new ArrayList[nPropagations];
				for(int i=0;i<nPropagations;i++)
					best_traceGeneration[i]=new ArrayList<Activation_Influence>(traces[i]);
			}
			
	
		}while(activeUsers.size()<nNodes && countTrials<maxTrials);
		
		traces=best_traceGeneration;
		
		System.out.println(" - - - - - - - - -");
		System.out.println("Covered users\t"+best_trace_Generation_score);
		double avg_lenght=0.0;
		for(int i=0;i<nPropagations;i++)
			avg_lenght+=traces[i].size();
		avg_lenght/=nPropagations;
		System.out.println("Average length of the traces\t"+avg_lenght);
		System.out.println();
		
	}//generate
	
	private HashSet<Integer> getUsersForTrace(
			ArrayList<Activation_Influence> arrayList) {
		HashSet<Integer> ris=new HashSet<Integer>();
		for(Activation_Influence a:arrayList){
			if(a.userId>=0)
				ris.add(a.userId);
		}	
		return ris;
	}

	private void store(String folder) {
		try {
			PrintWriter pw_act = new PrintWriter(""+folder+"/actionLog");
			pw_act.println("vertexId\ttraceId\ttimestamp\tInfluencer");
			
			
			PrintWriter traceCommunities=new PrintWriter(""+folder+"/traceCommunityList.txt");
						
			TreeSet<Integer>activeNodes=new TreeSet<Integer>();
			
			int nActivations=0;
			
			double avg_trace_lenght=0;
			double avg_communities=0;
			
			for (int i=0;i<traces.length;i++) {
				Collections.sort(traces[i]);
				avg_trace_lenght+=traces[i].size();
				TreeSet<Integer>communities=new TreeSet<Integer>();
				for(int j=0;j<traces[i].size();j++){
					Activation_Influence a=traces[i].get(j);
					if(a==null)
						throw new RuntimeException();
					Integer nodeId=a.userId;
					if(nodeId<0||nodeId==null)
						continue;
					nActivations++;
					activeNodes.add(a.userId);
					int ass=assignment.getCommunity(a.userId);
					communities.add(ass);
					int influencer =a.influencerId;
					String inf="-";
					if(influencer>=0)
						inf=""+influencer;
							
					pw_act.println(""+a.userId+"\t"+a.itemId+"\t"+a.timeStamp+"\t"+inf);
				}
				
				avg_communities+=communities.size();
				
				traceCommunities.print(i+"\t");
				for(int c:communities){
					traceCommunities.print((c+1)+" ");
				}
				traceCommunities.print("\n");
				
				
			}
			avg_trace_lenght/=nPropagations;
			avg_communities/=nPropagations;
			
			traceCommunities.close();
			pw_act.close();
			
			PrintWriter pw_netRate = new PrintWriter(""+folder+"/actionlog_net_rate");
			
			HashMap<Integer, Integer> node2Id=new HashMap<Integer, Integer>();
			
			
			int id=0;
			for(int node:activeNodes){
				node2Id.put(node, id);
				pw_netRate.println(id + "," + node);
				id++;
			}
			pw_netRate.println();
			long horizon = -1;
		
			for (int i = 0; i < traces.length; i++) {
				for (int j = 0; j < traces[i].size(); j++) {
					Activation a=traces[i].get(j);
					if(a.userId<0)continue;
					pw_netRate.print(node2Id.get(a.userId) + "," + a.timeStamp);
					
					
					if (a.timeStamp > horizon) {
						horizon = a.timeStamp;
					}
					if (j < traces[i].size() - 1) {
						pw_netRate.print(",");
					}
					
				}
				pw_netRate.println();	
			}
			pw_netRate.close();
			
			
			PrintWriter pw_ic = new PrintWriter(""+folder+"/actionlog_ic");
			
			String placeholder="omega";
			String start="omega";

			for(int i=0;i<traces.length;i++){
				ArrayList<Activation_Influence> activations_item=traces[i];
				Collections.sort(activations_item);
				
				pw_ic.println("\t"+start+"\t0");
				for(Activation a:activations_item){
					pw_ic.println(placeholder+"\t"+a.userId+"\t"+a.timeStamp);	
				}
			}
			
			
			pw_ic.flush();
			pw_ic.close();
			
			PrintWriter stat = new PrintWriter(""+folder+"/actionlog_statistics.txt");
			
			stat.println("#activeNodes\t"+activeNodes.size());
			
			stat.println("#activations\t"+nActivations);
			
			double avg_activation_nodes=(double)nActivations/activeNodes.size();
			
			stat.println("#traces\t"+traces.length);
		
			stat.println("Avg length of the traces\t"+avg_trace_lenght);
			stat.println("Avg #communities for trace\t"+avg_communities);
			stat.println("Avg activation 4 node\t"+avg_activation_nodes);
			
			stat.println("#horizon\t"+horizon);
			
			
			stat.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/*public static void main(String[]options) throws Exception{
		int nFold=5;
		String propModels[]=new String[]{"lt","nr"};
		String datasets[]=new String[]{"s1","s2","s3","s4"};
		double tM=0.2;
		int nTraces=1500;
		int minLenght=5;
		for(String dataset:datasets){
			
			for(String propModel:propModels){
				
				for(int fold=1;fold<=nFold;fold++){
					
					generate(dataset, propModel, fold, tM,minLenght,nTraces);
					
				}
			
			}
		}
	}
	*/
	
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void generate(String dataset,String propModel,int fold,double tM,int minL,int nTraces) throws Exception {
		PropagationGenerator pg=new PropagationGenerator();
		System.out.println("Propagation Generation");
		
		String []options=new String[]{"-net", "resources/datasets/synthetic/"+dataset+"/network.csv", 
				"-com","resources/datasets/synthetic/"+dataset+"/community.dat","-nprop",""+nTraces,"-propModel",propModel,"-communityMix",""+tM,"-minL",""+minL};
		
		
		String output="resources/datasets/synthetic/"+dataset+"/"+propModel+"/fold"+fold;
		
		if(options.length==0){
			printUsage();
			return;
		}
		
		for (int i = 0; i < options.length; i++) {
			if (options[i].equals("-help")) {
				printUsage();
				System.exit(-1);
			}
			if (options[i].equals("-net")) {
				pg.network = new LinksHandler();
				System.out.println("Setting network:\t" + options[i + 1]);
				pg.network.read(options[i + 1]);
				pg.network.printInfo();
				
				i++;
			}
			if (options[i].equals("-com")) {
				pg.assignment = new CommunityAssignmentHandler();
				System.out.println("Setting community assignment:\t" + options[i + 1]);
				pg.assignment.read(options[i + 1]);
				i++;
			}

			if (options[i].equals("-nprop")) {
				pg.nPropagations = Integer.parseInt(options[i + 1]);
				System.out.println("Setting #propagation:\t"
						+ pg.nPropagations);
				i++;
			}
			
			if (options[i].equals("-communityMix")) {
				pg.communityMix = Double.parseDouble(options[i + 1]);
				System.out.println("Setting community mix  probability:\t"
						+ pg.communityMix);
				i++;
			}
			
			if (options[i].equals("-propModel")) {
				String s=options[i + 1];
				
				if(s.equalsIgnoreCase("nr")){
					pg.pm=new IC_Rate_PropagationModel();
					System.out.println("Setting propagation model probability: NETRATE");
				}
				else
				
					if(s.equalsIgnoreCase("lt")){
						pg.pm=new LT_Discrete_PropagationModel();
						System.out.println("Setting propagation model probability: LT Discrete");
				}
				else{
					pg.pm=new IC_PropagationModel();
					System.out.println("Setting propagation model probability: IC");
				}
				i++;
			}
			
			if (options[i].equals("-minL")) {
				pg.minLenght = Integer.parseInt(options[i + 1]);
				System.out.println("Setting min trace lenght:\t" + pg.minLenght);
				i++;
			}
			if (options[i].equals("-seed")) {
				pg.seed = Integer.parseInt(options[i + 1]);
				System.out.println("Setting seed:\t" + pg.seed);
				i++;
			}
		}//options
		
		pg.generate();
		pg.store(output);
		System.out.println("DONE");
	}//main
	
	
	
	
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] options) throws Exception {
		PropagationGenerator pg=new PropagationGenerator();
		System.out.println("Propagation Generation");
		//0 0.05 0.1 0.15 0.2
		double tM=0.2;
		
		int fold=14;
		String dataset="s3";
		String propModel="nr";
		int nProp=50000;
		options=new String[]{"-net", "resources/datasets/synthetic/"+dataset+"/network.csv", 
				"-com","resources/datasets/synthetic/"+dataset+"/community.dat","-nprop",""+nProp,"-propModel",propModel,"-communityMix",""+tM};
		
		
		String output="resources/datasets/synthetic/"+dataset+"/"+propModel+"/fold"+fold;
		
		if(options.length==0){
			printUsage();
			return;
		}
		
		for (int i = 0; i < options.length; i++) {
			if (options[i].equals("-help")) {
				printUsage();
				System.exit(-1);
			}
			if (options[i].equals("-net")) {
				pg.network = new LinksHandler();
				System.out.println("Setting network:\t" + options[i + 1]);
				pg.network.read(options[i + 1]);
				pg.network.printInfo();
				
				i++;
			}
			if (options[i].equals("-com")) {
				pg.assignment = new CommunityAssignmentHandler();
				System.out.println("Setting community assignment:\t" + options[i + 1]);
				pg.assignment.read(options[i + 1]);
				i++;
			}

			if (options[i].equals("-nprop")) {
				pg.nPropagations = Integer.parseInt(options[i + 1]);
				System.out.println("Setting #propagation:\t"
						+ pg.nPropagations);
				i++;
			}
			
			if (options[i].equals("-communityMix")) {
				pg.communityMix = Double.parseDouble(options[i + 1]);
				System.out.println("Setting community mix  probability:\t"
						+ pg.communityMix);
				i++;
			}
			
			if (options[i].equals("-propModel")) {
				String s=options[i + 1];
				
				if(s.equalsIgnoreCase("nr")){
					pg.pm=new IC_Rate_PropagationModel();
					System.out.println("Setting propagation model probability: NETRATE");
				}
				else
				
					if(s.equalsIgnoreCase("lt")){
						pg.pm=new LT_Discrete_PropagationModel();
						System.out.println("Setting propagation model probability: LT");
				}
				else{
					pg.pm=new IC_PropagationModel();
					System.out.println("Setting propagation model probability: IC");
				}
				i++;
			}
			
			if (options[i].equals("-minL")) {
				pg.minLenght = Integer.parseInt(options[i + 1]);
				System.out.println("Setting delta:\t" + pg.minLenght);
				i++;
			}
			if (options[i].equals("-seed")) {
				pg.seed = Integer.parseInt(options[i + 1]);
				System.out.println("Setting seed:\t" + pg.seed);
				i++;
			}
		}//options
		
		pg.generate();
		pg.store(output);
		System.out.println("DONE");
	}//main

	
	protected static void printUsage() {
		System.out.println("\t-net: network file (default graph.txt)");
		System.out.println("\t-com: community assignment file");
		System.out.println("\t-nprop: number of propagations (default 1000)");
		System.out.println("\t-propModel: propagation model (IC|LT|NETRATE)  default IC");
		System.out.println("\t-communityMix: probability of next community  (default 0)");
		System.out.println("\t-minL: minimum lenght of transactions (default 5)");
		System.out.println("\t-seed: seed for randoms generator (default current time)");
		System.out.println("The program will produce 3 files in current folder:");
		System.out.println("\t1) activationLog.txt contains the list of the "
				+ "generated activations");
		System.out.println("\t2) traceCommunityList.txt contains for each "
				+ "propagation trace, a list of initial and all communities");
		System.out.println("\t3) netRateInput.txt NETRATE program input file");
		System.out.println("\t4) statistics.txt contains generation statistics");
	}
	
	
}//PropagationGenerator
