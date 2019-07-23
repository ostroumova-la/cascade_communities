package myRandBased;
public class Label  implements Comparable<Label>
{
	public int lablename = 0;
	public double Energy = 0.00;

	public Label(int i, double j)
	{
		lablename = i;
		Energy = j;



	}

	@Override
	public int compareTo(Label o) {
		
        return (Double.compare(this.Energy, o.Energy));
	}
	
}
