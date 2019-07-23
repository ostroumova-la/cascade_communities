package beans;

public class Feature {

	public int featureId;
	public int featureWeight;
	
	public Feature(int featureId, int weight) {
		this.featureId=featureId;
		this.featureWeight=weight;
	}
	
	public int getTagId() {
		return featureId;
	}
	
	public void setTagId(int tagId) {
		this.featureId = tagId;
	}
	
	public int getFeatureWeight() {
		return featureWeight;
	}
	
	public void setFeatureWeight(int weight) {
		this.featureWeight = weight;
	}

	@Override
	public String toString() {
		return "Feature [featureId=" + featureId + ", featureWeight=" + featureWeight + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + featureId;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Feature other = (Feature) obj;
		if (featureId != other.featureId)
			return false;
		return true;
	}
	

}
