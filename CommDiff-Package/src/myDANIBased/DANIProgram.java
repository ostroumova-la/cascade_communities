package myDANIBased;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.StringTokenizer;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;
import java.io.*;
import java.io.ObjectInputStream.GetField;
import javax.print.attribute.standard.OrientationRequested;
import evaluation.EvaluateMutualInformation;
public class DANIProgram
{
	private static BufferedReader lines;
	private static BufferedReader dCD;
    
	public static void main(String[] args) throws IOException
	{ //Inputs:
		int nodes=0;
		int[] Community;
		double[] WeightComm;
		double Totalh = 0.00;
		String Cascades="";
		String DANIOut="";
		String Output="";
		String Result="";
		int mm=0;
		
		
		double epsilon=0.08;
		int step_init=10;
		int step_size=5;
		int step_final=nodes;
		
		boolean detail=false;
		String groundtruth="";
		String detailoutput="";
		if (args.length == 0) {
			printUsage();
			return;
		}
		int step_final_check=0;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("--help")) {
				printUsage();
				return;
			}
			if (args[i].equals("-i")) {
				Cascades = args[i + 1];
				i++;
			}
			if (args[i].equals("-o")) {
				Output = args[i + 1];
				i++;
			}
			if (args[i].equals("-p")) {
				DANIOut = args[i + 1];
				i++;
			}
			if (args[i].equals("-r")) {
				Result = args[i + 1];
				i++;
			}
			if (args[i].equals("-mm")) {
				mm=Integer.parseInt(args[i + 1]);
				i++;
			}
			if (args[i].equals("-ep")) {
				epsilon=Double.parseDouble(args[i + 1]);
				i++;
			}
			
				
			if (args[i].equals("-si")) {
				step_init=Integer.parseInt(args[i + 1]);
				i++;
			}
			
			if (args[i].equals("-ss")) {
				step_size=Integer.parseInt(args[i + 1]);
				i++;
			}
			if (args[i].equals("-sf")) {
				step_final=Integer.parseInt(args[i + 1]);
				
				step_final_check=1;
				i++;
			}
			if (args[i].equals("-detail")) {
				detail=Boolean.parseBoolean(args[i + 1]);
				System.out.println("Details is "+detail);
				i++;
			}
			if (args[i].equals("-groundtruth")) {
				groundtruth=(args[i + 1]);
				i++;
			}
			if (args[i].equals("-detailoutput")) {
				detailoutput=(args[i + 1]);
				i++;
			}
			
		}
		long startTime = System.currentTimeMillis();
	if(mm==0){
		try {
			Process p = Runtime.getRuntime().exec("../DiffusionCommunity/main -i:"+Cascades+" -Y:"+DANIOut);
			BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String hhhh="";
			//If I remove this part, the shell script will not run!
			while ((hhhh=stdInput.readLine()) != null) {
				System.out.println(hhhh);
			}
		}
		catch (IOException e) {
		}
}
	System.out.println("DANI is done!");	
	String encoding = "UTF-8";
		dCD = new BufferedReader(new InputStreamReader(new FileInputStream(DANIOut), encoding));
			
		lines = new BufferedReader(new InputStreamReader(new FileInputStream(Cascades), encoding));
			
		BufferedWriter ResultCommunity = new BufferedWriter(new FileWriter(Output));
		FileWriter Log = new FileWriter(Result,true);
        java.util.ArrayList<Double> diff = new java.util.ArrayList<Double>();
		java.util.ArrayList<Node> Listnodes = new java.util.ArrayList<Node>();
		java.util.ArrayList<Node> Edges = new java.util.ArrayList<Node>();
		String line=null;
		while ((line = dCD.readLine()) != null)
		{
			String[] index = line.split("[\\t]", -1);
			Node M = new Node();
			M.name = index[0] + "->" + index[1];
			M.w = Double.parseDouble(index[2]);
			Edges.add(M);
			int check = 0;
			for (int i = 0; i < Listnodes.size(); i++)
			{
				if (Listnodes.get(i).name.equals(index[0]))
				{
					Listnodes.get(i).w = Listnodes.get(i).w + Double.parseDouble(index[2]);
					check = 1;
				}
			}
			if (check == 0)
			{
				Node N = new Node();
				N.name = index[0];
				N.w = Double.parseDouble(index[2]);
				Listnodes.add(N);
			}
		}
		//Sort
		Collections.sort(Listnodes);
		Collections.sort(Edges);
		//Cosine:
	
		Vector <Orginal_Node> Total=new Vector<Orginal_Node>();
		
		
		int column = 0; //number of Total cascades
		
		int checking = 0;
		line=null;
		//Reading Cascades from file
		while ((line = lines.readLine()) != null)
		{
			if (line.equals(""))
			{
				checking = 1;
				
				
				for(int i=0;i<nodes;i++){
					Total.add(new Orginal_Node());
					
				}
				continue;
			}
			if(checking ==0)
			{
               //number of nodes from first part of cascade file
				nodes++;
			}
			if (checking == 1)
			{
				column++;
				String[] index = line.split("[;]", -1);
				for (int i = 0; i < index.length; i++)
				{
					if(!(index[i].equals("")))
					{
					
					int avali= Integer.parseInt(index[i].split("[,]", -1)[0]);
					
				
					Total.get(avali).Own++;
					for(int j=i+1;j<index.length;j++)
					{
						int dovomi= Integer.parseInt(index[j].split("[,]", -1)[0]);	
						NE myhelpavali=Total.get(avali).Neighbour.get(dovomi);
						NE myhelpdovomi=Total.get(dovomi).Neighbour.get(avali);
				        if(myhelpavali!=null)
				        {
				        	myhelpavali.numCascades++;
				        }
				        else
				        	Total.get(avali).Neighbour.put(dovomi, new NE(1,0.00));
				        	
				        
				        if(myhelpdovomi!=null)
				        {
				        	myhelpdovomi.numCascades++;
				        }
				        else
				        {
				        
				        	Total.get(dovomi).Neighbour.put(avali, new NE(1,0.00));
				        }
						
					}
				}
			}
			}
		}
        
		
		int[] community_base = new int[nodes];
		int[] community_new = new int[nodes];
		double[] Modular = new double[nodes];
		Log.write(nodes+"\t"+column+"\t");
		System.out.println("Nodes\t"+nodes);
		System.out.println("number of cascades: " + column);
		
		
		
		
		for(int i=0;i<nodes;i++){
			if(Total.get(i).Own!=0){
			for (Entry<Integer, NE> e : Total.get(i).Neighbour.entrySet()) {
				double sorat=e.getValue().numCascades;
				
				double similarity = ((sorat) / (Math.sqrt(Total.get(i).Own) * Math.sqrt(Total.get(e.getKey()).Own)));
				e.getValue().CosineRank=similarity;
				Modular[i] += similarity;
                Totalh += (similarity);
				
			
				}
				
			}
		}
	
		System.out.println("Cosine finished");
		int numcom =0;
		java.util.ArrayList<Double> stop = new java.util.ArrayList<Double>();
	int preventbadcase=0;
	if (step_final_check == 0)
	{
	step_final=nodes;
	}
//Initialized the number of communities with step!
for (int step = step_init; step < step_final; step = step + step_size)
		
		{
	int iiii=0;
		
		 if(preventbadcase==1)
	{
	System.out.println("\n BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD \n");
		 break;
	}		
			if (stop.size() >= 3)
			{
				if ((stop.get(stop.size() - 2) >= stop.get(stop.size() - 3)) && (stop.get(stop.size() - 2) >= stop.get(stop.size() - 1)))
				{
					
					Log.write(step+"\t");
					break;
				}
			}
			
			numcom = step; //60 best
			
			Community = new int[nodes];
			WeightComm = new double[nodes];
			int count = 1;
			for (int i = Listnodes.size()-1;i >(Listnodes.size()-numcom); i--)
			{
				Community[(int) Long.parseLong(Listnodes.get(i).name)] = count;
				WeightComm[(int) Long.parseLong(Listnodes.get(i).name)] = 1;
				for (int j = Edges.size() - 1; j >= 0; j--)
				{
					if (Edges.get(j).name.contains(Listnodes.get(i).name))
					{
						String[] token = (Edges.get(j).name).split("[->]"); //token[0] & token[2]
						if (Listnodes.get(i).name.equals(token[0]))
						{
							if ((WeightComm[(int) Long.parseLong(token[2])] == 0) || (WeightComm[(int) Long.parseLong(token[2])] < Edges.get(j).w))
							{
								Community[(int) Long.parseLong(token[2])] = count;
								WeightComm[(int) Long.parseLong(token[2])] = Edges.get(j).w;
							}
						}
						else if (Listnodes.get(i).name.equals(token[2]))
						{
							if ((WeightComm[(int) Long.parseLong(token[0])] == 0) || (WeightComm[(int) Long.parseLong(token[0])] < Edges.get(j).w))
							{
								Community[(int) Long.parseLong(token[0])] = count;
								
								WeightComm[(int) Long.parseLong(token[0])] = Edges.get(j).w;
							}
						}
					}
				}
				count++;
			}
			for (int y = 0; y < nodes; y++)
			{
				community_base[y] = Community[y];
				community_new[y] = community_base[y];
			}
			double different = 0.00;
			double darsad=0.00;
			int temp=0;
			while (true)
			{
				temp++;
				if (diff.size() > 0)
				{
					if(((temp>=3)&&(darsad < epsilon)) || (temp>=20)) //maryam: 20
					{
					
						break;
					}
				}
				if (diff.size() > 0)
				{
	
					//copy from community_new to community_base !! here first is target second is source
					System.arraycopy(community_new, 0, community_base, 0, nodes);
				}
				for (int i = 0; i < nodes; i++)
				{
					int check = i;
					java.util.ArrayList<Label> Tracking = new java.util.ArrayList<Label>();
					for (Entry<Integer, NE> e : Total.get(i).Neighbour.entrySet()) {
						if (community_base[e.getKey()] != 0)
						{	
							int where = 0;
							for (int y = 0; y < Tracking.size(); y++)
							{
								if (Tracking.get(y).lablename == community_base[e.getKey()])
								{
									Tracking.get(y).Energy += ((e.getValue().CosineRank) - ((Modular[i] * Modular[e.getKey()]) / (Totalh)));
									where = 1;
									break;
								}
							}
							if (where == 0)
							{
								Tracking.add(new Label(community_base[e.getKey()], ((e.getValue().CosineRank) - ((Modular[i] * Modular[e.getKey()]) / (Totalh)))));
							}
						}
						//i++;
					}
					
					if (Tracking.size() > 0)
					{
						Collections.sort(Tracking);
						if (Tracking.get(Tracking.size() - 1).lablename != community_base[check]) //&& ((Tracking[Tracking.Count() - 1].Energy - Tracking[Tracking.Count() - 2].Energy)>0.01))
						{
							community_new[check] = Tracking.get(Tracking.size()- 1).lablename;
						}
					}
					Tracking.clear();
					
				}
			
				darsad = DANIProgram.complementaryofPurity(community_base, community_new, numcom, nodes);
				
				different = DANIProgram.MutualInformation(community_base, community_new, numcom, nodes);
		
                if(different==-1)
                {
                	preventbadcase=1;
                	break;
                }
				diff.add(different);
				if (detail) //(accuracy or NMI) vs. iteration plot
				{
					printcommunity(community_base,Integer.toString(step)+" "+Integer.toString(iiii),detailoutput,nodes,numcom);
			
				
							iiii++;
						/*ArrayList<Integer>[] groundComm;
						try {
							groundComm = evaluation.EvaluateMutualInformation.maryamreadCommunities(groundtruth);
							ArrayList<Integer>[]InferredComm=ProductCommunities(community_new,nodes);
							EvaluateMutualInformation.maryam_evaluate(InferredComm, groundComm, Integer.toString(iiii), Integer.toString(step));
							
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}*/
					
					
					
			
					
					
				}
			}
			//Console.WriteLine("@@@@@@@@@@@@@@\t" + different);
 // stop is storing NMI for finding local maximum
			stop.add(different);
		diff.clear();
		}
		int[] nameofcommunity = new int[numcom];
		int InffredCommunity=0;
		for (int p = 0; p < nodes; p++)
		{
			int u = community_base[p];
			if (Integer.compare(nameofcommunity[u], 0)==0)
			{
				nameofcommunity[u] = 1;
				int ccc = 0;
				for (int y = 0; y < nodes; y++)
				{
					if (Integer.compare(community_base[y] , u)==0)
					{
						if (Integer.compare(ccc,0)==0)
						{
							ResultCommunity.write(y+"");
							//							System.out.print(y);
						}
						else
						{
							ResultCommunity.write(" "+ y);
							//							System.out.print(" "+y);
						}
						ccc = 1;
					}
				}
				if (Integer.compare(ccc,1)==0)
				{
					ResultCommunity.write("\n");
					InffredCommunity++;
					//					System.out.println();
				}
			}
		}
		ResultCommunity.close();
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Time:\t"+totalTime);
		Log.write(InffredCommunity+"\t"+totalTime+"\n");
		Log.close();
	}
	//////////////////////////////////// Function ///////////////////////////////////////////////////////
	public static void printcommunity(int[] arr,String name,String address,int nodes,int numcom) throws IOException{
		FileWriter myoutput = new FileWriter(address+name,false);
		int[] nameofcommunity = new int[numcom];
		for (int p = 0; p < nodes; p++)
		{
			int u = arr[p];
			if (Integer.compare(nameofcommunity[u], 0)==0)
			{
				nameofcommunity[u] = 1;
				int ccc = 0;
				for (int y = 0; y < nodes; y++)
				{
					if (Integer.compare(arr[y] , u)==0)
					{
						if (Integer.compare(ccc,0)==0)
						{
							myoutput.write(y+"");
						}
						else
						{
							myoutput.write(" "+ y);
						}
						ccc = 1;
					}
				}
				if (Integer.compare(ccc,1)==0)
				{
					myoutput.write("\n");
				
				}
			}
		}
		myoutput.close();
	}
	
	 public static int CountDistinctElements(int[] arr){
         int count=0;
	        for(int i=0;i<arr.length;i++){
	            boolean isDistinct = false;
	            for(int j=0;j<i;j++){
	                if(arr[i] == arr[j]){
	                    isDistinct = true;
	                    break;
	                }
	            }
	            if(!isDistinct){
	                count++;
	               
	            }
	        }
	        return count;
	    }
	 
		public static ArrayList<Integer>[] ProductCommunities(int[] arr,int nodes){
			int numofcom=CountDistinctElements(arr);
			ArrayList<Integer>[] ris= new ArrayList[numofcom];
			for(int c=0;c<numofcom;c++)
				ris[c]=new ArrayList<Integer>();
			
			int[] nameofcommunity = new int[numofcom];
			
			int index=0;
			for (int p = 0; p < nodes; p++)
			{
				int u = arr[p];
				if (Integer.compare(nameofcommunity[u], 0)==0)
				{
					nameofcommunity[u] = 1;
				
					for (int y = 0; y < nodes; y++)
					{
						if (Integer.compare(arr[y] , u)==0)
						{
								ris[index].add(y);	
							
							
						}
					}
					
						index++;
					
				}
			}
			
			return ris;
		}
	public static double MutualInformation(int[] c1, int[] c2, int NumOfCommunity, int nodes) throws IOException
	{
		double eps = 0.0000000001;
		double I = 0.0;
		double N = nodes;
		java.util.ArrayList <Integer> groupC1 = new java.util.ArrayList<Integer>();
		java.util.ArrayList <Integer> groupC2 = new java.util.ArrayList<Integer>();
		for (int k = 1;k <= NumOfCommunity;k++)
		{
			for (int i = 0;i < nodes;i++)
			{
				if (c1[i]== k )
				{
					groupC1.add(i);
				}
			}
			if (groupC1.size() == 0)
			{
				continue;
			}
			for (int h = 1;h <= NumOfCommunity;h++)
			{
				for (int j = 0;j < nodes;j++)
				{
					if (c2[j] == h)
					{
						groupC2.add(j);
					}
				}
				if (groupC2.size() == nodes)
				{
              return -1;
				}	
				
				if (groupC2.size() == 0)
				{
					continue;
				}
				//  Console.WriteLine("hi");
				int x_size = groupC1.size();
				int y_size = groupC2.size();
				double p_x = (double)x_size / N;
				double p_y = (double)y_size / N;
				int count = 0;
				for (int ss = 0;ss < groupC1.size();ss++)
				{
					for (int zz = 0;zz < groupC2.size();zz++)
					{
						if (Integer.compare(groupC1.get(ss),groupC2.get(zz))==0)
						{
							count++;
						}
					}
				}
				//****************************************
				double pxy = (count / N) + eps;
				I += pxy * Math.log(pxy / (p_x * p_y));
				//Check.write(p_x + "\t" + p_y + "\t" + pxy + "\t" + Math.log(pxy / (p_x * p_y)) + "\t" + I+"\n");
				groupC2.clear();
				//groupC2.shrink_to_fit();
			}
			groupC1.clear();
			//		groupC1.shrink_to_fit();
		}
		return I;
	}
	
	
	public static double complementaryofPurity(int[] c1, int[] c2, int NumOfCommunity, int nodes) throws IOException
	{
		//double eps=Double.MIN_VALUE;
		int TotalSim=0;
		
		double I = 0.0;
		double N = nodes;
		java.util.ArrayList <Integer> groupC1 = new java.util.ArrayList<Integer>();
		java.util.ArrayList <Integer> groupC2 = new java.util.ArrayList<Integer>();
		for (int k = 1;k <= NumOfCommunity;k++)
		{
			for (int i = 0;i < nodes;i++)
			{
				if (c2[i]== k )
				{
					groupC1.add(i);
				}
			}
			if (groupC1.size() == 0)
			{
				continue;
			}
			int valuemax=0;
			for (int h = 1;h <= NumOfCommunity;h++)
			{
				for (int j = 0;j < nodes;j++)
				{
					if (c1[j] == h)
					{
						groupC2.add(j);
					}
				}
				if (groupC2.size() == nodes)
				{
              				return -1;
				}	
				if (groupC2.size() == 0)
				{
					continue;
				}
	
				int count = 0;
				for (int ss = 0;ss < groupC1.size();ss++)
				{
					for (int zz = 0;zz < groupC2.size();zz++)
					{
						if (Integer.compare(groupC1.get(ss),groupC2.get(zz))==0)
						{
							count++;
						}
					}
				}
//				double pxy = (count / N) f0.05+ eps;
//				I += pxy * Math.log(pxy / (p_x * p_y));
				
				if(count>valuemax)
					valuemax=count;
				groupC2.clear();
			}
			TotalSim+=valuemax;
			
			groupC1.clear();
		}
		//System.out.println("Change:\t"+(N-TotalSim));
		I=((N-TotalSim)/N);
		return I;
	}
	private static void printUsage() {
		System.out
		.println("-i <cascade> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities>  -n <networkFile> ");
	}
}