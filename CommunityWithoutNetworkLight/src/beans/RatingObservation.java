package beans;

public class RatingObservation extends Activation {
	double rating;
	
	public RatingObservation(int userId, int itemId,double rating) {
		super(userId, itemId);
		this.rating=rating;
	}
	
	public RatingObservation(int userId, int itemId, long timeStamp,double rating) {
        super(userId, itemId,timeStamp);
        this.rating = rating;
    }

	
	public double getRating(){
		return rating;
	}
	
	public void setRating(double d){
		this.rating=d;
	}
	
	 @Override
	    public String toString() {
	        return "(" + userId + ", " + itemId + ", " + timeStamp + ","+rating+  ")";
	    }
}
