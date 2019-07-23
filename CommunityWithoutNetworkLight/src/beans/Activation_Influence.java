package beans;

public class Activation_Influence extends Activation {

	public int influencerId;
	
	public Activation_Influence(int userId, int itemId, long timeStamp) {
		super(userId, itemId, timeStamp);
		influencerId=-1;
	}
	
	
	public Activation_Influence(int userId, int itemId, long timeStamp,int influencerId) {
		super(userId, itemId, timeStamp);
		this.influencerId=influencerId;
	}


	
	public int getInfluencerId() {
		return influencerId;
	}


	public void setInfluencerId(int influencerId) {
		this.influencerId = influencerId;
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
