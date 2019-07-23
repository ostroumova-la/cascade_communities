package evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;

public class Evaluation_Utils {
	
	public static HashMap<Integer, Integer>readAssignments(String file)throws Exception{
		//Edited by Maryam
		System.out.println("Reading assignments file\t"+file);
		HashSet<Integer>communities=new HashSet<Integer>();
		HashMap<Integer, Integer>ris=new HashMap<Integer, Integer>();
		BufferedReader br=new BufferedReader(new FileReader(file));
		String line=br.readLine();
		
		
		StringTokenizer st;
		int community=0;
		while(line!=null){
			community++;
            st=new StringTokenizer(line);
            int over=st.countTokens();
			for(int h=0;h<over;h++){
			int node=Integer.parseInt(st.nextToken());
			communities.add(community);
			ris.put(node, community);
			}
			line=br.readLine();
		}
		br.close();
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
		int community=0;
		while(line!=null){
//			st=new StringTokenizer(line);
//			st.nextToken();
//			int community=Integer.parseInt(st.nextToken());
			community++;
			communities.add(community);
			line=br.readLine();
		}
		
		nCommunities=communities.size();
		br.close();
		

		
		@SuppressWarnings("unchecked")
		ArrayList<Integer>[] comm=new ArrayList[nCommunities];
		for(int c=0;c<nCommunities;c++)
			comm[c]=new ArrayList<Integer>();
		
		
		br=new BufferedReader(new FileReader(file));
		line=br.readLine();
		
		
		community=0;
		while(line!=null){
			community++;
            st=new StringTokenizer(line);
            int over=st.countTokens();
			for(int h=0;h<over;h++){
			int node=Integer.parseInt(st.nextToken());
			comm[community-1].add(node);
			}
			line=br.readLine();
		}
		br.close();
	
		
		
		return comm;	
	}//readCommunities
		

}
