package propagationGenerator;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import beans.Activation;
import handlers.ActivationsHandler;

public class ConvertToNetRateFormat {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("Convert Traces to Netrate Format");

		String confFile = null;
		String actionsFile = null;
		
		
		if (args.length == 0) {
			printUsage();
			return;
		}
		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("--help")) {
				printUsage();
				return;
			}
			if (args[i].equals("-a")) {
				actionsFile = args[i + 1];
				i++;
			}
			if (args[i].equals("-c")) {
				confFile = args[i + 1];
				i++;
			}
		
		}// for each arg
		
		
		ActivationsHandler p = new ActivationsHandler();
		p.read(actionsFile, confFile);
		p.printInfo();
		
		int nItems=p.getNItems();
		int[]itemArray=p.getItemArray();
		ArrayList<Activation>[] traces= new ArrayList[nItems];
		for (int i = 0; i < nItems; i++) {
			traces[i] = p.getActionsForItem(itemArray[i]);
			Collections.sort(traces[i]);
		}
		
		

		
		PrintWriter pw_netRate = new PrintWriter("netRateInput.txt");
		
		HashMap<Integer, Integer> node2Id=new HashMap<Integer, Integer>();
		
		
		int id=0;
		for(int node:p.getUserIndex().keySet()){
			node2Id.put(node, id);
			pw_netRate.println(id + "," + node);
			id++;
		}
		pw_netRate.println();
		long horizon = -1;
	
		
		for (int i = 0; i < traces.length; i++) {
			for (int j = 0; j < traces[i].size(); j++) {
				Activation a=traces[i].get(j);
				pw_netRate.print(node2Id.get(a.userId) + "," + a.timeStamp);
				
				if (a.timeStamp > horizon) {
					horizon = a.timeStamp;
				}
				if (j < traces[i].size() - 1) {
					pw_netRate.print(",");
				}
				
			}
			pw_netRate.println();	
		}
		pw_netRate.close();
		
		System.out.println("Horizon\t"+horizon);

	}//main

	
	
	private static void printUsage() {
		System.out.println("-a <actionlog> -c <readingConfFile> ");
	}//printUsage
	
}//args
