package myRandBased;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;
import java.io.*;
import java.io.ObjectInputStream.GetField;
import javax.print.attribute.standard.OrientationRequested;
import evaluation.EvaluateMutualInformation;
public class RandProgram
{
	private static BufferedReader lines;
	public static void main(String[] args) throws IOException
	{ // Inputs:
		int nodes = 0;
		double Totalh = 0.00;
		String Cascades = "";
		String Output = "";
		String Result = "";
		double epsilon=0.08;
		int step_init=10;
		int step_size=5;
		int step_final=nodes;
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
			}
			if (args[i].equals("-r")) {
				Result = args[i + 1];
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
		}
		long startTime = System.currentTimeMillis();
		lines = new BufferedReader(new FileReader(Cascades));
		BufferedWriter ResultCommunity = new BufferedWriter(new FileWriter(
				Output));
		FileWriter Log = new FileWriter(Result, true);
		java.util.ArrayList<Double> diff = new java.util.ArrayList<Double>();
		// java.util.HashMap<Integer,java.util.ArrayList<Integer>> Cosine = new
		// java.util.HashMap<Integer,java.util.ArrayList<Integer>>();
		// Cosine:
		Vector<Orginal_Node> Total = new Vector<Orginal_Node>();
		int column = 0; // number of Total cascades
		int checking = 0;
		String line = null;
		// Reading Cascades from file
		while ((line = lines.readLine()) != null)
		{
			if (line.equals(""))
			{
				checking = 1;
				for (int i = 0; i < nodes; i++) {
					Total.add(new Orginal_Node());
				}
				continue;
			}
			if (checking == 0)
			{
				// number of nodes from first part of cascade file
				nodes++;
			}
			if (checking == 1)
			{
				column++;
				String[] index = line.split("[;]", -1);
				for (int i = 0; i < index.length; i++)
				{
//System.out.println(column+"\t"+index[i]);
					int avali = Integer.parseInt(index[i].split("[,]", -1)[0]);
					Total.get(avali).Own++;
					for (int j = i + 1; j < index.length; j++) {
						int dovomi = Integer
								.parseInt(index[j].split("[,]", -1)[0]);
						NE myhelpavali = Total.get(avali).Neighbour.get(dovomi);
						NE myhelpdovomi = Total.get(dovomi).Neighbour
								.get(avali);
						if (myhelpavali != null) {
							myhelpavali.numCascades++;
						} else
							Total.get(avali).Neighbour.put(dovomi, new NE(1,
									0.00));
						if (myhelpdovomi != null) {
							myhelpdovomi.numCascades++;
						} else {
							Total.get(dovomi).Neighbour.put(avali, new NE(1,
									0.00));
						}
					}
				}
				//System.out.println(column+"\t is done!");
			}
		}
		int[] community_base = new int[nodes];
		int[] community_new = new int[nodes];
		double[] Modular = new double[nodes];
		Log.write(nodes + "\t" + column + "\t");
		System.out.println("Nodes\t" + nodes);
		System.out.println("number of cascades: " + column);
		for (int i = 0; i < nodes; i++) {
			if (Total.get(i).Own != 0) {
				for (Entry<Integer, NE> e : Total.get(i).Neighbour.entrySet()) {
					double sorat = e.getValue().numCascades;
					double similarity = ((sorat) / (Math.sqrt(Total.get(i).Own) * Math
							.sqrt(Total.get(e.getKey()).Own)));
					e.getValue().CosineRank = similarity;
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
//	for (int step = 2; step < 5; step = step + 1)
//for (int step = 10; step < nodes; step = step + 10)
for (int step = step_init; step < step_final; step = step + step_size)
		
		{
		
		 if(preventbadcase==1) break;
			System.out.println("Step is \t"+step);
			if (stop.size() >= 3)
			{
				if ((stop.get(stop.size() - 2) >= stop.get(stop.size() - 3))
						&& (stop.get(stop.size() - 2) >= stop
								.get(stop.size() - 1)))
				{
					// System.out.println("Here break:\t" + (step));
					Log.write(step + "\t");
					break;
				}
			}
			numcom = step; // 60 best
			// Console.WriteLine("***** Step:\t"+numcom);
			Random ran = new Random();
			for (int y = 0; y < nodes; y++)
			{
				// Check.write(Community[y]+"\t"+WeightComm[y]+"\n");
				community_base[y] = ran.nextInt(numcom);
				community_new[y] = community_base[y];
			}
			double different = 0.00;
			double darsad=0.00;
			int temp=0;
			System.out.println("nodes\t"+nodes);
			while (true)
			{
				temp++;
				if (diff.size() > 0)
				{
				//	double tafrigh = diff.get(diff.size() - 1) - diff.get(diff.size() - 2);
					
					System.out.println("darsad\t"+darsad);
				//	if (tafrigh < 0.000000000001)
			//	double epsilon=0.05 ;//Math.pow(Math.log10(Math.log10(nodes)),3.00);
					if(((temp>=3)&&(darsad < epsilon)) || (temp>=20))
					{
					
				//	if(temp==20)
						
                   System.out.println("Converge\n**********");
						break;
					}
				}
				if (diff.size() > 0)
				{
					// Console.WriteLine("here");
					// copy from community_new to community_base !! here first
					// is target second is source
					System.arraycopy(community_new, 0, community_base, 0, nodes);
				}
				for (int i = 0; i < nodes; i++)
				{
					int check = i;
					java.util.ArrayList<Label> Tracking = new java.util.ArrayList<Label>();
					for (Entry<Integer, NE> e : Total.get(i).Neighbour
							.entrySet()) {
						if (community_base[e.getKey()] != 0)
						{
							int where = 0;
							for (int y = 0; y < Tracking.size(); y++)
							{
								if (Tracking.get(y).lablename == community_base[e
										.getKey()])
								{
									Tracking.get(y).Energy += ((e.getValue().CosineRank) - ((Modular[i] * Modular[e
											.getKey()]) / (Totalh)));
									where = 1;
									break;
								}
							}
							if (where == 0)
							{
								Tracking.add(new Label(
										community_base[e.getKey()],
										((e.getValue().CosineRank) - ((Modular[i] * Modular[e
												.getKey()]) / (Totalh)))));
							}
						}
						// i++;
					}
					// Tracking.Sort((l, r) => l.Energy.compareTo(r.Energy));
					if (Tracking.size() > 0)
					{
						Collections.sort(Tracking);
						if (Tracking.get(Tracking.size() - 1).lablename != community_base[check]) // &&
																									// ((Tracking[Tracking.Count()
																									// -
																									// 1].Energy
																									// -
																									// Tracking[Tracking.Count()
																									// -
																									// 2].Energy)>0.01))
						{
							community_new[check] = Tracking.get(Tracking.size() - 1).lablename;
						}
					}
					Tracking.clear();
				}
				darsad = RandProgram.complementaryofPurity(community_base, community_new, numcom, nodes);
				different = RandProgram.MutualInformation(community_base,community_new, numcom, nodes);
				//System.out.println(different);
				if(different==-1)
                {
                	preventbadcase=1;
                	break;
                }
				diff.add(different);
			}
			// Console.WriteLine("@@@@@@@@@@@@@@\t" + different);
			stop.add(different);
			diff.clear();
		}
		int[] nameofcommunity = new int[numcom];
		int InffredCommunity = 0;
		for (int p = 0; p < nodes; p++)
		{
			int u = community_base[p];
			if (Integer.compare(nameofcommunity[u], 0) == 0)
			{
				nameofcommunity[u] = 1;
				int ccc = 0;
				for (int y = 0; y < nodes; y++)
				{
					if (Integer.compare(community_base[y], u) == 0)
					{
						if (Integer.compare(ccc, 0) == 0)
						{
							ResultCommunity.write(y + "");
							// System.out.print(y);
						}
						else
						{
							ResultCommunity.write(" " + y);
							// System.out.print(" "+y);
						}
						ccc = 1;
					}
				}
				if (Integer.compare(ccc, 1) == 0)
				{
					ResultCommunity.write("\n");
					InffredCommunity++;
					// System.out.println();
				}
			}
		}
		ResultCommunity.close();
		long endTime = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Time:\t" + totalTime);
		Log.write(InffredCommunity + "\t" + totalTime + "\n");
		Log.close();
	
	}
	//////////////////////////////////// Function ///////////////////////////////////////////////////////
	public static double MutualInformation(int[] c1, int[] c2, int NumOfCommunity, int nodes) throws IOException
	{
		//double eps=Double.MIN_VALUE;
		double eps = 0.0000000001;
		//for (int r = 0; r < nodes; r++)
		//{
		//Check.write(r + "-> " + c1[r] + "\t" + c2[r]+"\n");
		//}
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
				if (c1[i]== k )
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
//				double pxy = (count / N) + eps;
//				I += pxy * Math.log(pxy / (p_x * p_y));
				
				if(count>valuemax)
					valuemax=count;
				groupC2.clear();
			}
			TotalSim+=valuemax;
			groupC1.clear();
		}
System.out.println("Change:\t"+(N-TotalSim));
		I=((N-TotalSim)/N);
		return I;
		
	}
	private static void printUsage() {
		System.out
		.println("-i <cascade> -c <confFile> -k <nCommunities> -o <output> -maxIt <maxIt> -g <groundTruthCommunities>  -n <networkFile> ");
	}
}