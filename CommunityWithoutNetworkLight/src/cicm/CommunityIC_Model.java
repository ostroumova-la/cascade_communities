package cicm;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

import java.io.Serializable;

public class CommunityIC_Model  implements Serializable{

	
	private static final long serialVersionUID = 1L;
	Integer users[];
	Int2IntOpenHashMap userIdToIndex;

	double gamma[][];
	double influence_probabilities[][];
	double priors[];
	int nCommunities;
	
	
	public CommunityIC_Model(){
		
	}
	

	
	public Integer[] getUsers() {
		return users;
	}



	public void setUsers(Integer[] users) {
		this.users = users;
	}





	public Int2IntOpenHashMap getUserIdToIndex() {
		return userIdToIndex;
	}



	public void setUserIdToIndex(Int2IntOpenHashMap userIdToIndex) {
		this.userIdToIndex = userIdToIndex;
	}





	public double[][] getGamma() {
		return gamma;
	}





	public void setGamma(double[][] gamma) {
		this.gamma = gamma;
	}





	public double[][] getInfluenceProbabilities() {
		return influence_probabilities;
	}





	public void setInfluence_weights(double[][] influence_weights) {
		this.influence_probabilities = influence_weights;
	}



	public double[] getPriors() {
		return priors;
	}





	public void setPriors(double[] priors) {
		this.priors = priors;
	}



	public int getnCommunities() {
		return nCommunities;
	}





	public void setnCommunities(int nCommunities) {
		this.nCommunities = nCommunities;
	}




	public static long getSerialversionuid() {
		return serialVersionUID;
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}//CIC model
