package myRandBased;
public class Sim  implements Comparable<Sim>

{
	public int node1 = -1;
	public int node2 = -1;
	public double w = 0.00;
	public Sim(int i, int j, double v)
	{
		node1 = i;
		node2 = j;
		w = v;


	}

	@Override
	public int compareTo(Sim o) {
	return (Integer.compare(this.node1, o.node1));	
	}
}
