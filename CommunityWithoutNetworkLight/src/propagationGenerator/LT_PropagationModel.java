package propagationGenerator;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import beans.Activation;
import beans.Activation_Influence;
import beans.Edge;

public class LT_PropagationModel extends PropagationModel {

	double thresholds[];
	Int2IntOpenHashMap nodeId2Index;

	@Override
	public void initialize() {
		nodeId2Index=network.getVertexId2Index();		
		infProb=new HashMap<Edge, Double>();
		for(Edge e: network.getEdges()){
				double indegree=network.getInlinksForVertex(e.destination).size();
				Double inf_value=1.0/indegree;
				infProb.put(e,inf_value);
		}		
		thresholds=new double[vertexSet.length];
		
		for(int i=0;i<vertexSet.length;i++){
			thresholds[i]=rand.nextDouble();
		}
	}//initialize
	
	
	@Override
	public ArrayList<Activation_Influence> generateTrace(HashSet<Integer> seedNodes,int traceId) {
		
		
		HashSet<Integer> active_nodes=new HashSet<Integer>(seedNodes);
		HashSet<Integer> candidates=new HashSet<Integer>();
		
		HashSet<Integer>next_candidates;
		
		ArrayList<Activation_Influence> trace=new ArrayList<Activation_Influence>();
		
		ArrayList<Activation> all_previous_activations=new ArrayList<Activation>();
		
		for(int seedNode:seedNodes)
			all_previous_activations.add(new Activation(seedNode, traceId,0));
		
		HashMap<Integer, ArrayList<Activation>> new_Activations=new HashMap<Integer, ArrayList<Activation>>();
		
		HashSet<Integer>active_nodes_copy=new HashSet<Integer>(active_nodes);
		Edge e;
		for(int activeNode:active_nodes_copy){
			for(int candidate:network.getOutLinksForVertex(activeNode)){
				if(!active_nodes.contains(candidate)){
					e=new Edge(activeNode,candidate);
					if(rand.nextDouble()<infProb.get(e)){
						active_nodes.add(candidate);
					//	long timestamp=(long)(rand.nextExp(gamma.get(e))*1000);
						long timestam=rand.nextInt(1000);
						trace.add(new Activation_Influence(candidate, traceId, timestam,activeNode));
					}
				}
			}//for each candidate 
		}//for each seed node
		
		
		
		for(int active_node:active_nodes){
			if(active_node>0)
				candidates.addAll(network.getOutLinksForVertex(active_node));
		}
		candidates.removeAll(active_nodes);
		
		
		
		long t=1000;
		while(candidates.size()>0){
			next_candidates=new HashSet<Integer>();
			new_Activations.clear();
			
			for(int candidateNode:candidates){
				int candidateIndex=nodeId2Index.get(candidateNode);
				double cum_inf=0.0;
				
				ArrayList<Activation>prev=new ArrayList<Activation>();
				
				for(Activation a:trace){
					int active_node=a.userId;
					if(network.existsEdge(active_node, candidateNode)){
						cum_inf+=infProb.get(new Edge(active_node,candidateNode));
						prev.add(a);
					}
				}
				
				if(cum_inf>=thresholds[candidateIndex]){
					//become active
					new_Activations.put(candidateNode, prev);
				}//become active
				
			}//for each candidate
			
			
			for(int new_active_node:new_Activations.keySet()){
				
				long offsetTime=rand.nextInt(1000);
				long timestamp=t+offsetTime;
			
				Activation_Influence a=new Activation_Influence(new_active_node, traceId,timestamp);
				trace.add(a);
				all_previous_activations.add(a);
			
				if(active_nodes.contains(new_active_node))
					throw new RuntimeException();
				
				active_nodes.add(new_active_node);
				
				
			}//for each new active node
			//add candidates
			for(int new_active_node:new_Activations.keySet())
				for(int neighbor:network.getOutLinksForVertex(new_active_node)){
					if(!active_nodes.contains(neighbor)){
						next_candidates.add(neighbor);
					}
				}	
			
			candidates=next_candidates;
			t+=1000;
		}//while
		
		return trace;		
	}

	


}//LT_PropagationModel
