package propagationGenerator;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import beans.Activation;
import beans.Activation_Influence;
import beans.Edge;

public class IC_Rate_PropagationModel extends PropagationModel {

	

	
	@Override
	public void initialize() {
		initializeGamma();
	
		int maxOutDegree=-1;
		for(int nodeId:network.getVertexArray()){
			if(nodeId>0){
				int outdegree=network.getOutLinksForVertex(nodeId).size();
				if(maxOutDegree<outdegree)
					maxOutDegree=outdegree;
			}
		}
		
		
		infProb=new HashMap<Edge, Double>();
		
		
		for(Edge e: network.getEdges()){
			if(e.source<0){
				infProb.put(e,rand.nextUniform(sigmaProb_min,sigmaProb_max));
			}
			else{

				
				int outdegree_u=network.getOutLinksForVertex(e.source).size();
			
				double indegree_v=network.getInlinksForVertex(e.destination).size();
				
				double inf_prob_topology=outdegree_u/(maxOutDegree+delta) * 1d/indegree_v;
				Double inf_value= a* (inf_prob_topology) +(1-a)*rand.nextUniform(2*sigmaProb_max,1.0);
				infProb.put(e,inf_value);
			}
			

		}//
	}//initialize

	
	@Override
	public ArrayList<Activation_Influence> generateTrace(HashSet<Integer> seedNodes,int traceId) {
		
		HashSet<Activation_Influence>current_activations=new HashSet<Activation_Influence>();
		
		for(int seedNode:seedNodes){
			current_activations.add(new Activation_Influence(seedNode, traceId,0));
		}
	
		
		HashSet<Activation_Influence>next_activations;
		
		HashSet<Integer> all_activeNodes=new HashSet<Integer>(seedNodes);
		
		
		HashSet<Edge> trials=new HashSet<Edge>();
		
		
		ArrayList<Activation_Influence> trace=new ArrayList<Activation_Influence>();
		
	
		HashMap<Integer, ArrayList<Activation_Influence>> successful_trials=new HashMap<Integer, ArrayList<Activation_Influence>>();
		
		ArrayList<Activation_Influence>succ_inf;
		
		Edge e;
	
		
		while(current_activations.size()>0){
			next_activations=new HashSet<Activation_Influence>();
			successful_trials=new HashMap<Integer, ArrayList<Activation_Influence>>();
			
			for(Activation_Influence a:current_activations){
				
				
				int activeNode=a.userId;
				
				
				for(int candidate:network.getOutLinksForVertex(activeNode)){
					e=new Edge(activeNode,candidate);
					if(!all_activeNodes.contains(candidate) && !trials.contains(e)){
						
						trials.add(e);
						
					
						if(rand.nextDouble()<infProb.get(e)){
							//the node becomes active
							
							//add to successful trials
							
							succ_inf=successful_trials.get(candidate);
							if(succ_inf==null)
								succ_inf=new ArrayList<Activation_Influence>();
							if(succ_inf.contains(a)){
								throw new RuntimeException();
							}
							succ_inf.add(a);
							successful_trials.put(candidate, succ_inf);

						}//becomes active
					}//if
					
				}//for each candidate
				
				
				
			}//for each current activation
			

			
			for(int new_activeNode:successful_trials.keySet()){
				succ_inf=successful_trials.get(new_activeNode);

				
				if(all_activeNodes.contains(new_activeNode)){
					System.out.println(""+new_activeNode);
					throw new RuntimeException();
				}
				
		
				
				long activationTimestamp=Long.MAX_VALUE;
			
				int influencer=-1;
				for(Activation prev:succ_inf){
					
					Double gamma_a=gamma.get(new Edge(prev.userId,new_activeNode));

					long timestamp=(long)(prev.timeStamp+rand.nextExp(gamma_a)*1000);
					
					if (timestamp<activationTimestamp){
						activationTimestamp=timestamp;
						influencer=prev.userId;
					}
				}
				
				Activation_Influence a=new Activation_Influence(new_activeNode, traceId,activationTimestamp,influencer);
				trace.add(a);
				next_activations.add(a);
				
				all_activeNodes.add(new_activeNode);
			}
			
			
			
			current_activations=next_activations;
			
			
		}//while
		
	
		return trace;
	}//generate
	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}



	

}//IC_PropagationModel
