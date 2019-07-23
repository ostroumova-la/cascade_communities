package myRandBased;
public class Node implements Comparable<Node>
  {
		 public String name = "";
		 public double w = 0.00;

			@Override
			public int compareTo(Node o) {
			
		        return (Double.compare(this.w, o.w));
			}
			
			
			
			
	
  }
