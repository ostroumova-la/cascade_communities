package propagationGenerator;

import handlers.LinksHandler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import util.Randoms;
import beans.Activation_Influence;
import beans.Edge;

public abstract class PropagationModel {

	LinksHandler network;	
	int []vertexSet;
	static Randoms rand=new Randoms(System.currentTimeMillis());
	
	HashMap<Edge,Double> infProb;
	
	HashMap<Edge,Double> gamma;
	
	protected static double sigmaProb_max=0.05;
	protected static double sigmaProb_min=0.02;


	double a=0.9;
	
	double delta=0.0;

	static double gamma_shape=2;
	static double gamma_scale=0.3;
	
	public abstract void initialize();
	
	
	public void initializeGamma(){
		gamma=new HashMap<Edge,Double> ();
		for(Edge e:network.getEdges()){
			double gamma_e=rand.nextGamma(gamma_shape,gamma_scale);
			gamma.put(e, gamma_e);
		}
	}//initializeGamma
	

	
	public abstract ArrayList<Activation_Influence> generateTrace(HashSet<Integer> activeNodes,int traceId);
	
	public void setNetwork(LinksHandler network) {
		this.network=network;
		this.vertexSet=network.getVertexArray();
	}
	
	
	
}//PropagationModel
