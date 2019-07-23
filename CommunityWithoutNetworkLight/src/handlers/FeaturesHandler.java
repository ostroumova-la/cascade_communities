package handlers;

import beans.Feature;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;

public class FeaturesHandler {

	HashMap<Integer, ArrayList<Feature>> itemIdToFeatures;
	HashMap<Integer,ArrayList<Integer>>featureIdToItems;
	HashMap<Integer,String>featureIdToName;
	
	HashMap<Integer,String>itemIdToTitle;

	public FeaturesHandler(){
		itemIdToFeatures=new HashMap<Integer, ArrayList<Feature>>();
		featureIdToItems=new HashMap<Integer, ArrayList<Integer>>();
		featureIdToName=new HashMap<Integer, String>();
		itemIdToTitle=new HashMap<Integer, String>();
	}
	
	public void readFeaturesAssociations(String fileName) throws IOException{
        System.out.println("------------------------------------------------");
        System.out.println("Reading features from: " + fileName);
       
        itemIdToFeatures.clear();
        featureIdToItems.clear();
        
       
        BufferedReader br = new BufferedReader(new FileReader(
                new File(fileName)));
        String line = br.readLine();
        System.out.println("Header: " + line);
		line=br.readLine();
		StringTokenizer st;
		int itemId;
		int feature;
		int weight;
		ArrayList<Feature>list;
		ArrayList<Integer>list_items;

		int count=0;
		while(line!=null){
			st=new StringTokenizer(line);
			itemId=Integer.parseInt(st.nextToken());
			feature=Integer.parseInt(st.nextToken());
			
			if(st.hasMoreTokens())
			weight=Integer.parseInt(st.nextToken());
			else
				weight=1;
			
			list=itemIdToFeatures.get(itemId);
			if(list==null)
				list=new ArrayList<Feature>();
			list.add(new Feature(feature,weight));
			itemIdToFeatures.put(itemId, list);
			
			list_items=featureIdToItems.get(feature);
			if(list_items==null)
				list_items=new ArrayList<Integer>();
			list_items.add(itemId);
			featureIdToItems.put(feature, list_items);
			
			line=br.readLine();
			count++;
		}
		System.out.println("Reading feature file:DONE "+count+ " lines");        
	}//read
	
	
	public void readFeatureNames(String fileName)throws Exception{
		
        System.out.println("------------------------------------------------");
        System.out.println("Reading features from: " + fileName);
       
		
        featureIdToName.clear();
        
        
        BufferedReader br = new BufferedReader(new FileReader(
                new File(fileName)));
        String line = br.readLine();
        System.out.println("Header: " + line);
		line=br.readLine();
		StringTokenizer st;
		int itemId;
		String feature;
		
		while(line!=null){
			st=new StringTokenizer(line);
			itemId=Integer.parseInt(st.nextToken());
			feature=line.substring(line.indexOf("\t")).trim();
			featureIdToName.put(itemId, feature);
			line=br.readLine();
		}
	
		System.out.println("Reading feature file:DONE");        
	}
	
	
	public void readItemTitles(String fileName)throws Exception{
		  System.out.println("------------------------------------------------");
	        System.out.println("Reading titles from: " + fileName);
	       
			
	        itemIdToTitle.clear();
	        
	        
	        BufferedReader br = new BufferedReader(new FileReader(
	                new File(fileName)));
	        String line = br.readLine();
	        System.out.println("Header: " + line);
			line=br.readLine();
			StringTokenizer st;
			int itemId;
			String title;
			
			while(line!=null){
				st=new StringTokenizer(line);
				itemId=Integer.parseInt(st.nextToken());
				title=line.substring(line.indexOf("\t")).trim();
				itemIdToTitle.put(itemId, title);
				line=br.readLine();
			}
		
			System.out.println("Reading names file:DONE"); 
	}//readItemNames
	
	
	public String getTitle(int itemId){
		return itemIdToTitle.get(itemId);
	}
	
	public Integer[]getFeatureSet(){
		return featureIdToItems.keySet().toArray(new Integer[featureIdToItems.size()]);
	}
	
	public HashSet<Integer>getItemSet(){
		return new HashSet<Integer>(itemIdToFeatures.keySet());
	}
	
	
	public ArrayList<Feature> getFeatures(int itemId){
		return itemIdToFeatures.get(itemId);
	}
	
	
	public ArrayList<Integer> getItemsForFeature(int feature){
		return featureIdToItems.get(feature);
	}
	
	
	public String getTagName(int tag){
		return featureIdToName.get(tag);
	}
	
	public int getNFeatures(){
		return featureIdToItems.size();
	}
	
	
	public void printInfo(){
		
		double avgItemsFortag=0.0;
		double avgTagsForItem=0.0;
		int maxTagsForItem=-1;
		int minTagsForItem=Integer.MAX_VALUE;
		
		int minItemsForTag=Integer.MAX_VALUE;
		int maxItemsForTag=-1;

		
		for(int itemId: itemIdToFeatures.keySet()){
			int size=itemIdToFeatures.get(itemId).size();
			avgTagsForItem+=size;
			if(size>maxTagsForItem)
				maxTagsForItem=size;
			if(size<minTagsForItem)
				minTagsForItem=size;
		}
		
		avgTagsForItem/=itemIdToFeatures.size();
		
		for(int tag:featureIdToItems.keySet()){
			int size=featureIdToItems.get(tag).size();
			if(size>maxItemsForTag)
				maxItemsForTag=size;
			if(size<minItemsForTag)
				minItemsForTag=size;
			avgItemsFortag+=size;		
		}
		avgItemsFortag/=featureIdToItems.size();
		
        System.out.println("------------------------------------------------");
        System.out.println(". . . . . . . . Feature info . . . . . . . .");
        System.out.println("#Items \t"+itemIdToFeatures.size());
        System.out.println("#Features \t"+featureIdToItems.size());
        System.out.println();
        System.out.println("Avg #Features item \t" + avgTagsForItem);
        System.out.println("Min #Features item \t" + minTagsForItem);
        System.out.println("Max #Features item \t" + maxTagsForItem);
        System.out.println();
        System.out.println("Avg #item for Feature \t" + avgItemsFortag);
        System.out.println("Min #item for Feature \t" + minItemsForTag);
        System.out.println("Max #item for Feature \t" + maxItemsForTag);
        System.out.println("------------------------------------------------");

		
	}//printInfo
	
	
	public void printStatistics(String folder)throws Exception{
		
		PrintWriter pw= new PrintWriter(new FileWriter(folder+"Tag2Items"));
		pw.println("FeatureId\tNItems");
		
		for(int tag:featureIdToItems.keySet()){
			int size=featureIdToItems.get(tag).size();
			pw.println(""+tag+"\t"+size);
		}
		pw.flush();
		pw.close();
		
		pw= new PrintWriter(new FileWriter(folder+"Item2Tags"));
		pw.println("ItemId\tNtags");
		
		for(int itemId:itemIdToFeatures.keySet()){
			int size=itemIdToFeatures.get(itemId).size();
			pw.println(""+itemId+"\t"+size);
		}
		pw.flush();
		pw.close();
	}
	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		String flixsterTags="resources/datasets/lastFm/tagAnnotations";
		FeaturesHandler th=new FeaturesHandler();
		th.readFeaturesAssociations(flixsterTags);
		th.printInfo();
	
		//th.printStatistics("resources/datasets/lastFm/");
	}

}//TagHandler
