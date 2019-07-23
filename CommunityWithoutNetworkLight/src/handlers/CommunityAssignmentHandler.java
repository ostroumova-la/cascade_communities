package handlers;

import it.unimi.dsi.fastutil.ints.*;

import java.io.*;
import java.util.*;

public class CommunityAssignmentHandler {
	private HashMap<Integer, Integer> vertex2Community;
	private ArrayList<Integer>[] communities;
	private String header;
	private int nCommunities;

	public String separator;
	public boolean skip_first;
	public int first_community_id, index_of_user_id, index_of_comm_id;

	public CommunityAssignmentHandler() {
		this(null);
	}

	public CommunityAssignmentHandler(String config) {
		processConfigFile(config);
		vertex2Community = new HashMap<Integer, Integer>();
	}

	private void processConfigFile(String readConfigFile) {
		try {
			Properties properties = new Properties();
			properties.load(new FileInputStream(readConfigFile));
			this.index_of_user_id = Integer.parseInt(properties
					.getProperty("index_of_user_id"));
			this.index_of_comm_id = Integer.parseInt(properties
					.getProperty("index_of_comm_id"));
			this.first_community_id = Integer.parseInt(properties
					.getProperty("first_community_id"));
			this.skip_first = properties.getProperty("skip_first")
					.equalsIgnoreCase("true");
			this.separator = properties.getProperty("separator").trim();
			if (separator.equalsIgnoreCase("tab")) {
				this.separator = "\t";
			}
		} catch (Throwable e) {
			default_init();
		}
	}

	private void default_init() {
		this.skip_first = false;
		this.first_community_id = 1;
		this.index_of_user_id = 0;
		this.index_of_comm_id = 1;
		this.separator = "\t";
	}

	public void read(String inputFile) throws IOException {
		System.out.println("------------------------------------------------");
		System.out.println("Reading communities from " + inputFile);
		BufferedReader br = new BufferedReader(new FileReader(inputFile));
		IntOpenHashSet communities_set = new IntOpenHashSet();
		if (skip_first) {
			System.out.println("Header: " + (header = br.readLine()));
		}
		String line = null;
		int vertexId = 1;
		Int2IntOpenHashMap communityId2Index = new Int2IntOpenHashMap();
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(separator);
			if (index_of_user_id != -1) {
				vertexId = Integer.parseInt(tokens[index_of_user_id].trim());
			}
			int communityId = Integer.parseInt(tokens[index_of_comm_id].trim())
					- first_community_id;
			communities_set.add(communityId);
			vertex2Community.put(vertexId, communityId);
			vertexId++;
		}
		br.close();
		nCommunities = communities_set.size();
		int[] communities_array = communities_set.toIntArray();
		Arrays.sort(communities_array);
		communities = new ArrayList[nCommunities];
		for (int i = 0; i < nCommunities; i++) {
			communities[i] = new ArrayList<Integer>();
			communityId2Index.put(communities_array[i], i);
		}
		for (int nodeId : vertex2Community.keySet()) {
			int communityId = vertex2Community.get(nodeId);
			int communityIndex = communityId2Index.get(communityId);
			vertex2Community.put(nodeId, communityIndex);
			communities[communityIndex].add(nodeId);
		}
		System.out.println("Communities loaded!");	
		
		System.out.println("------------------------------------------------");
	}

	public void printInfo() {
		System.out.println("************ CONFIGURATION ************");
		System.out.println("Setting skip_first_line\t\t= " + skip_first);
		System.out.println("Setting index_of_user_id\t= " + index_of_user_id);
		System.out.println("Setting index_of_comm_id\t= " + index_of_comm_id);
		System.out.println("Setting first_community_id\t= "
				+ first_community_id);
		System.out.println("Setting separator\t\t= "
				+ (separator.equals("\t") ? "tab" : separator));
		System.out.println("***************************************");
		System.out.println("*********** COMMUNITIES INFOS *********");
		if (skip_first) {
			System.out.println("Header\t= " + header);
		}
		System.out.println("K\t= " + nCommunities);
		System.out.println("***************************************");
	}

	public HashMap<Integer, Integer> getVertex2Community() {
		return vertex2Community;
	}

	public void setVertex2Community(HashMap<Integer, Integer> vertex2Community) {
		this.vertex2Community = vertex2Community;
	}

	public ArrayList<Integer>[] getCommunities() {
		return communities;
	}

	public void setCommunities(ArrayList<Integer>[] communities) {
		this.communities = communities;
	}

	public int getnCommunities() {
		return nCommunities;
	}

	public void setnCommunities(int nCommunities) {
		this.nCommunities = nCommunities;
	}

	public void setHeader(String header) {
		this.header = header;
	}

	public int getCommunity(int node){
		return vertex2Community.get(node);
	}
	
	public String getHeader() {
		return header;
	}
}
