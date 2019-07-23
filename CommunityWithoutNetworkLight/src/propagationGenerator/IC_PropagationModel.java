package propagationGenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import util.Randoms;

import beans.Activation;
import beans.Activation_Influence;
import beans.Edge;

public class IC_PropagationModel extends PropagationModel {

	
	@Override
	public void initialize() {
		infProb=new HashMap<Edge, Double>();
		for(Edge e: network.getEdges()){
			if(e.source<0){
				infProb.put(e,rand.nextUniform(sigmaProb_min,sigmaProb_max));
			}
			else{
				double indegree=network.getInlinksForVertex(e.destination).size();
				Double inf_value=1.0/indegree;
				infProb.put(e,inf_value);
			}	
		}//for each edge
	}//initialize

	
	@Override
	public ArrayList<Activation_Influence> generateTrace(HashSet<Integer> seedNodes,int traceId) {
		
		HashSet<Integer>current_active_nodes=new HashSet<Integer>(seedNodes);
		HashSet<Integer>next_active_nodes;
		HashSet<Integer> all_activeNodes=new HashSet<Integer>(seedNodes);
		
		ArrayList<Activation_Influence> trace=new ArrayList<Activation_Influence>();
		
		for(int seedNode:seedNodes){
			trace.add(new Activation_Influence(seedNode, traceId,0));
		}
	
		
		long t=0;
		while(current_active_nodes.size()>0){
			next_active_nodes=new HashSet<Integer>();
			for(int activeNode:current_active_nodes){
				for(int candidate:network.getOutLinksForVertex(activeNode)){
					if(!all_activeNodes.contains(candidate)){
						if(rand.nextDouble()<infProb.get(new Edge(activeNode,candidate))){
							//the node becomes active
							long offsetTime=rand.nextInt(1000);
							
							trace.add(new Activation_Influence(candidate,traceId,t+offsetTime));
							next_active_nodes.add(candidate);
							all_activeNodes.add(candidate);
						}//becomes active
					}//if
					
				}//for each candidate
				
			}
			current_active_nodes=next_active_nodes;
			t+=1000;
		}//while
		
		// TODO Auto-generated method stub
		return trace;
	}//generate
	


}//IC_PropagationModel
