package myRandBased;
public class NE{
	
	public int numCascades=0; //number of cascades that the two nodes taked part in
	public double CosineRank=0.00; //the cosine similarity of two nodes
	public NE(int i, double j)
	{
		numCascades = i;
		CosineRank = j;
	}
	public NE()
	{
		numCascades = 0;
		CosineRank = 0.00;
	}
}
