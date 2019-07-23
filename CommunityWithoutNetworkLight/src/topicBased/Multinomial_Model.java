package topicBased;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.io.Serializable;

public class Multinomial_Model implements Serializable{

	
	int nVertices;
	Integer[] vertexSet;
	Int2IntOpenHashMap traceId2Index;
	Int2IntOpenHashMap vertexId2Index;
	int nTraces;
	Integer[] traceSet;

	double pi[];
	double gamma[][];
	double phi[][];
	double K;
	
	public Multinomial_Model(){}
	
	
	
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



	public double[][] getGamma() {
		return gamma;
	}



	public void setGamma(double[][] gamma) {
		this.gamma = gamma;
	}



	public double[][] getPhi() {
		return phi;
	}



	public void setPhi(double[][] phi) {
		this.phi = phi;
	}



	public double getK() {
		return K;
	}



	public void setK(double k) {
		K = k;
	}

	
	public double[] getPi() {
		return pi;
	}



	public void setPi(double[] pi) {
		this.pi = pi;
	}

	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
