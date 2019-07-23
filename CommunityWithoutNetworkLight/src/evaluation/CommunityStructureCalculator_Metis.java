package evaluation;

import handlers.LinksHandler;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import utils.ArrayUtilities;

public class CommunityStructureCalculator_Metis {

	
	public static void main(String[] args) throws Exception {
		System.out.println(" - - - - Community Structure Evaluator FOR METIS- - - - ");
		
		if(args.length==0){
			printUsage();
			return;	
		}
		
		String networkPath=null;
		String metis_communities=null;
		boolean directed=false;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("--help")) {
				printUsage();
				return;
			}
			
			if (args[i].equals("-n")) {
				networkPath = args[i + 1];
				i++;
			}
			
			
			if (args[i].equals("-c")) {
				metis_communities= args[i + 1];
				i++;
			}

			if (args[i].equals("-d")) {
				String s=args[i+1];
				if(s.equalsIgnoreCase("y")||s.equalsIgnoreCase("yes")||s.equalsIgnoreCase("true")  )
					directed=true;
				i++;
			}	
			
			
		}// for each args
		
		String errors="";
		
		if(networkPath==null)
			errors="No Network file has been provided\n";
		
		if(metis_communities.length()==0)
			errors+="No community file has been provided.";
		
		if(errors.length()>0){
			errors="Errors log:\n"+ errors;
			System.out.println(errors);
			return;
		}
	
		
		LinksHandler network = new LinksHandler();
		network.read(networkPath);
		network.printInfo();

		
		int users_set[]=network.getVertexArray();
		
		
		HashMap<Integer, Integer>assignments=new HashMap<Integer, Integer>();
		
		
		BufferedReader br=new BufferedReader(new FileReader(metis_communities));
		HashSet<Integer> set_communities=new HashSet<Integer>();
		
		int node_index=0;
		for(String line=br.readLine();line!=null;line=br.readLine()){
			int community=Integer.parseInt(line.trim())+1;	
			assignments.put(users_set[node_index], community);
			set_communities.add(community);
			node_index++;
		}
		ArrayList<Integer>[] communities=new ArrayList[set_communities.size()];
		for(int i=0;i<set_communities.size();i++)
			communities[i]=new ArrayList<Integer>();
		
		
		for(int node_id:assignments.keySet()){
			int c=assignments.get(node_id)-1;
			communities[c].add(node_id);
		}
		
				
		System.out.println("Assignments\t"+assignments.size());
		
		CommunityStructureCalculator csc = new CommunityStructureCalculator(
				network, assignments,communities);
		
	
		System.out.println(" - - - - - - Results (K = "+ communities.length + ") - - - - - - - ");
		
		
		double modularity=0.0;
		
		double[] conductance;
		double []internalDensity;
		double []cutratio;
		
		if(directed){
			modularity=csc.computeModularityDirectGraph();
			conductance=csc.computeConductanceDirectedGraph();
			internalDensity=csc.computeInternalDensityDirectedGraph();
			cutratio=csc.computeCutRatioDirectedGraph();
		}
		else{
			modularity=csc.computeModularity();
			conductance=csc.computeConductance();
			internalDensity=csc.computeInternalDensity();
			cutratio=csc.computeCutRatio();
		}
		
		double entropy=csc.computeEntropy();
		
		double sizes[]=csc.getCommunitiesSize();
		
		System.out.println(" - - - Modularity - - - ");
		printForExcel(modularity);
		System.out.println();
		System.out.println(" - - - Conductance - - - - ");
		System.out.println("Min\t"+ArrayUtilities.findMin(conductance)[1]);
		
		System.out.println("Max\t"+ArrayUtilities.findMax(conductance)[1]);
		
		System.out.println("Avg\t"+ArrayUtilities.avg(conductance));
	
		System.out.println("harmonic mean\t"+ArrayUtilities.HarmonicMean(conductance));
		System.out.println("Median\t"+ArrayUtilities.median(conductance));
		
		System.out.println();
		System.out.println("- - - Internal Density - - - ");
		System.out.println("Min\t"+ArrayUtilities.findMin(internalDensity)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(internalDensity)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(internalDensity));
		System.out.println("harmonic mean\t"+ArrayUtilities.HarmonicMean(internalDensity));
		System.out.println("Median\t"+ArrayUtilities.median(internalDensity));
		
		System.out.println("- - - Cut Ratio- - - ");
		System.out.println("Min\t"+ArrayUtilities.findMin(cutratio)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(cutratio)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(cutratio));
		System.out.println("harmonic mean\t"+ArrayUtilities.HarmonicMean(cutratio));
		System.out.println("Median\t"+ArrayUtilities.median(cutratio));
		System.out.println();
		
		System.out.println("- - - Sizes- - - ");
		System.out.println("Min\t"+ArrayUtilities.findMin(sizes)[1]);
		System.out.println("Max\t"+ArrayUtilities.findMax(sizes)[1]);
		System.out.println("Avg\t"+ArrayUtilities.avg(sizes));
		System.out.println("harmonic mean\t"+ArrayUtilities.HarmonicMean(sizes));
		System.out.println("Median\t"+ArrayUtilities.median(sizes));
		System.out.println();
		
		System.out.println("Entropy\t"+entropy);

		
	}

	private static void printForExcel(double v) {
		System.out.println(String.valueOf(v).replace('.', ','));
	}

	private static void printUsage() {
		System.out.println("-n <networkFile> -c <metis_output>  -d <directed(yes|no)>");
	}
	
}
	
