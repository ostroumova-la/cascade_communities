package myRandBased;
import java.util.TreeMap;
public class Orginal_Node {

public int Own=0; //count all the cascade this node had taken part in
public TreeMap<Integer,NE> Neighbour;
public Orginal_Node(){
	Own=0;
	Neighbour = new TreeMap<Integer,NE>();
}

}
		

