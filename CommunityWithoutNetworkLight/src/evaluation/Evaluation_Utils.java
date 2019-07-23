package evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;

public class Evaluation_Utils {
	

	
	public static HashMap<Integer, Integer>readAssignments(String file)throws Exception{
		System.out.println("Reading assignments file\t"+file);
		HashSet<Integer>communities=new HashSet<Integer>();
		HashMap<Integer, Integer>ris=new HashMap<Integer, Integer>();
		BufferedReader br=new BufferedReader(new FileReader(file));
		String line=br.readLine();
		
		
		StringTokenizer st;
		while(line!=null){
			st=new StringTokenizer(line);
			int node=Integer.parseInt(st.nextToken());
			int community=Integer.parseInt(st.nextToken());
			communities.add(community);
			ris.put(node, community);
			line=br.readLine();
		}
		
		System.out.println("Done");
		System.out.println("N Communities\t"+communities.size());
		System.out.println("N Nodes\t"+ris.keySet().size());
		return ris;
	}//readAssignments
	
	
	public static ArrayList<Integer>[] readCommunities(String file)throws Exception{
		int nCommunities=-1;
		HashSet<Integer> communities=new HashSet<Integer>();
		BufferedReader br=new BufferedReader(new FileReader(file));
		String line=br.readLine();
		
		StringTokenizer st;
		while(line!=null){
			st=new StringTokenizer(line);
			st.nextToken();
			int community=Integer.parseInt(st.nextToken());
			communities.add(community);
			line=br.readLine();
		}
		
		nCommunities=communities.size();
		

		
		ArrayList<Integer>[] comm=new ArrayList[nCommunities];
		for(int c=0;c<nCommunities;c++)
			comm[c]=new ArrayList<Integer>();
		
		
		br=new BufferedReader(new FileReader(file));
		line=br.readLine();
		while(line!=null){
			st=new StringTokenizer(line);
			int node=Integer.parseInt(st.nextToken());
			int community=Integer.parseInt(st.nextToken());
			comm[community-1].add(node);
			line=br.readLine();
		}
		
		return comm;	
	}//readCommunities
		

}
