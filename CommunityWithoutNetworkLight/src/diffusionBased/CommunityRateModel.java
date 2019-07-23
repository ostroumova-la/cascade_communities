package diffusionBased;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.io.Serializable;

public class CommunityRateModel implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int nVertices;
	Integer[] vertexSet;
	Int2IntOpenHashMap traceId2Index;
	Int2IntOpenHashMap vertexId2Index;
	int nTraces;
	Integer[] traceSet;
	
	double alpha[][];
	double gamma[][];
	int communityAssignments[];

	
	public CommunityRateModel(){}
	
	
	public double[][] getGamma(){
		return gamma;
	}
	
	public void setGamma(double [][]g){
		this.gamma=g;
	}
	
	public int getnVertices() {
		return nVertices;
	}



	public void setnVertices(int nVertices) {
		this.nVertices = nVertices;
	}



	public Integer[] getVertexSet() {
		return vertexSet;
	}



	public void setVertexSet(Integer[] vertexSet) {
		this.vertexSet = vertexSet;
	}



	public Int2IntOpenHashMap getTraceId2Index() {
		return traceId2Index;
	}



	public void setTraceId2Index(Int2IntOpenHashMap traceId2Index) {
		this.traceId2Index = traceId2Index;
	}



	public Int2IntOpenHashMap getVertexId2Index() {
		return vertexId2Index;
	}



	public void setVertexId2Index(Int2IntOpenHashMap vertexId2Index) {
		this.vertexId2Index = vertexId2Index;
	}



	public int getnTraces() {
		return nTraces;
	}



	public void setnTraces(int nTraces) {
		this.nTraces = nTraces;
	}



	public Integer[] getTraceSet() {
		return traceSet;
	}



	public void setTraceSet(Integer[] traceSet) {
		this.traceSet = traceSet;
	}
	
	
	public double [][] getAlpha() {
		return alpha;
	}

	public void setAlpha(double[][] alpha) {
		this.alpha = alpha;
	}

	public int[] getCommunityAssignments() {
		return communityAssignments;
	}

	public void setCommunityAssignments(int[] communityAssignments) {
		this.communityAssignments = communityAssignments;
	}


}//CommunityRateModel

