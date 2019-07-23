package util;

import handlers.*;
import it.unimi.dsi.fastutil.ints.*;

import java.io.*;

public class MetisFormatConverter {

	public void convert(String networkFile, String outputFile)
			throws IOException {
		LinksHandler network = new LinksHandler();
		network.read(networkFile);
		network.printInfo();
		int[] users = network.getVertexArray();
		Int2IntOpenHashMap name2Index = network.getVertexId2Index();
		System.out.print("Writing network in metis format...");
		PrintWriter pw = new PrintWriter(outputFile);
		pw.println(network.getNVertices() + " " + network.getNUnDirectedEdges());
		for (int i = 0; i < users.length; i++) {
			IntArrayList out_links = network.getOutLinksForVertex(users[i]);
			IntArrayList in_links = network.getInlinksForVertex(users[i]);
			IntOpenHashSet neighbours = new IntOpenHashSet();
			for (int out : out_links) {
				neighbours.add(name2Index.get(out) + 1);
			}
			for (int in : in_links) {
				neighbours.add(name2Index.get(in) + 1);
			}
			for (int n : neighbours) {
				pw.print(n + " ");
			}
			if (neighbours.size() > 0)
				pw.println();
		}
		pw.close();
		System.out.println("done!");
	}

	public static void main(String[] args) {
		try {
	//		args = new String[] {
	//				"resources/datasets/synthetic/LTM_500/network.dat",
		//			"network_metis.txt" };
			System.out.println(MetisFormatConverter.class.getSimpleName());
			if (args.length < 2) {
				printUsage();
				return;
			}
			MetisFormatConverter mfc = new MetisFormatConverter();
			mfc.convert(args[0], args[1]);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void printUsage() {
		System.out.println("<inputFilePath> <outputFilePath>");
	}
}