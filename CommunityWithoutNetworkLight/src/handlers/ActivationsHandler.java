package handlers;

import it.unimi.dsi.fastutil.ints.*;

import java.io.*;
import java.text.*;
import java.util.*;

import beans.*;

public class ActivationsHandler {

    public enum TimeFormat {
        None, UnixTimestamp, Millis, Date;
    }

    private ArrayList<Activation> actions;
    private TreeSet<Integer> userSet, itemSet;
    private int nUsers, nItems;
    private HashMap<UserItemPair, Integer> directIndex;
    // mapping
    private Int2IntOpenHashMap itemId2Index, userId2Index;
    // userIndex[u] = indici delle azioni compiute da u
    private IntArrayList userIndex[];
    // itemIndex[i] = indici delle azioni in cui viene usato i
    private IntArrayList itemIndex[];
    // configurazione
    public TimeFormat timeFormat=TimeFormat.UnixTimestamp;
    public int index_of_user_id=0, index_of_item_id=1, index_of_timestamp=2, index_of_rating=-1;
    public String separator="\t", header;
	public boolean skip_first_line=false;
	public SimpleDateFormat dfm;

	TreeSet<Double>rating_scale;
	
    public ActivationsHandler() {
        userSet = new TreeSet<Integer>();
        itemSet = new TreeSet<Integer>();
        actions = new ArrayList<Activation>();
        rating_scale=new TreeSet<Double>();
    }

    public ActivationsHandler(TreeSet<Integer> users, TreeSet<Integer> items,
            ArrayList<Activation> actions) {
        userSet = users;
        itemSet = items;
        this.actions = actions;
        this.nUsers = users.size();
        this.nItems = items.size();
    }
    

    
    private void processConfigFile(String readConfigFile) throws IOException{
        Properties properties = new Properties();
        properties.load(new FileInputStream(readConfigFile));
        index_of_user_id = Integer.parseInt(properties.getProperty("index_of_user_id"));
        index_of_item_id = Integer.parseInt(properties.getProperty("index_of_item_id"));
        index_of_timestamp = Integer.parseInt(properties.getProperty("index_of_timestamp"));
        index_of_rating=-1;
        String indexRating_s=properties.getProperty("index_of_rating");
        if(indexRating_s!=null && indexRating_s.length()>0)
        	index_of_rating = Integer.parseInt(indexRating_s);

        this.skip_first_line = properties.getProperty("skip_first_line")
                .equalsIgnoreCase("true");
        
        
        this.dfm = null;
       

        String time_f = null;
        if (index_of_timestamp >= 0) {
            time_f = properties.getProperty("timestamp_format").trim();
            if (time_f.equalsIgnoreCase("None")) {
                this.timeFormat = TimeFormat.None;
            } else if (time_f.equalsIgnoreCase("unix_timestamp")) {
                this.timeFormat = TimeFormat.UnixTimestamp;
            } else if (time_f.equalsIgnoreCase("millisecs")) {
                this.timeFormat = TimeFormat.Millis;
            } else {
                this.timeFormat = TimeFormat.Date;
                dfm = new SimpleDateFormat(time_f);
            }
        }
        this.separator = properties.getProperty("separator").trim();
        if (separator.equalsIgnoreCase("tab"))
            this.separator = "\t";
        
        System.out.println("Index User\t"+index_of_user_id);
        System.out.println("Index Item\t"+index_of_item_id);
        System.out.println("Index Timestap\t"+index_of_timestamp);
        System.out.println("Index rating\t"+index_of_rating);
        System.out.println("Separator\t"+separator);

   
    }
    

    
    
    public void read(String dataFile, String readConfigFile) throws Exception {
        System.out.println("------------------------------------------------");
        System.out.println("Reading actions from file: " + dataFile);
       
        if(readConfigFile!=null)
        	processConfigFile(readConfigFile);
        
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        String line = br.readLine();
        header = "";
        if (skip_first_line) {
            header = line.toString();
            line = br.readLine();
        }
        String[] tokens;
        StringTokenizer st;
        if(header.length()>0){
	        System.out.println("Header: " + header);
	        st = new StringTokenizer(header, separator);
	       tokens = new String[st.countTokens()];
        }
        else
        {
        	 st = new StringTokenizer(line, separator);
        	 tokens = new String[st.countTokens()];
        }
        Activation po = null;
        int lineIndex=0;
        rating_scale.clear();
        userSet.clear();
        itemSet.clear();
        while (line != null) {
        	
        	lineIndex++;
            try{
	        	st = new StringTokenizer(line,separator);
	            
	        	int index = 0;
	         
	            while (st.hasMoreTokens()) {
	                tokens[index] = st.nextToken();
	                index++;
	            }
	            int userId = Integer.parseInt(tokens[index_of_user_id].trim());
	            int itemId = Integer.parseInt(tokens[index_of_item_id].trim());
	            long timeStamp = 0L;
	            Double rating;
	            if (index_of_timestamp >= 0) {
	                if (timeFormat == TimeFormat.Date) {
	                    timeStamp = dfm.parse(tokens[index_of_timestamp].trim()).getTime() / 1000;
	               
	                } else
	                    timeStamp = Long.parseLong(tokens[index_of_timestamp]);
	                if (timeFormat == TimeFormat.Millis) {
	                    timeStamp /= 1000;
	                }
	            }
	            if(index_of_rating>=0){
	            	rating=Double.parseDouble(tokens[index_of_rating].trim());
	            	if(Double.isNaN(rating)){
	            		throw new RuntimeException(""+line);
	            	}
	            	po=new RatingObservation(userId, itemId,timeStamp ,rating);
	            	rating_scale.add(rating);
	            }
	            else
	            	po = new Activation(userId, itemId, timeStamp);
	            
	            if (po != null) {
	                actions.add(po);
	                userSet.add(userId);
	                itemSet.add(itemId);
	            }
        	}catch(Exception e){
        		System.out.println("Error in processing the line "+lineIndex);
        		System.out.println(line);
        		System.out.println("Tokens");
        		for(int i=0;i<tokens.length;i++)
        			System.out.print(tokens[i]+"\t");
        		System.out.println();
        		e.printStackTrace();
        		return;
        	}
            line = br.readLine();
        }
        nUsers = userSet.size();
        nItems = itemSet.size();
        System.out.println("Dataset loaded!");
        System.out.println("------------------------------------------------");
        buildIndex();
    }
    
    
    public void read(String dataFile) throws Exception {
        System.out.println("------------------------------------------------");
        System.out.println("Reading actions from file: " + dataFile);
       
 /*       index_of_user_id=0;
        index_of_item_id=1;
        index_of_timestamp=2;
        index_of_rating=-1;
        skip_first_line=false;*/
        //separator="\t";
        		
        
        
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        String line = br.readLine();
        header = "";
        if (skip_first_line) {
            header = line.toString();
            line = br.readLine();
        }
        String[] tokens;
        StringTokenizer st;
        if(header.length()>0){
	        System.out.println("Header: " + header);
	        st = new StringTokenizer(header, separator);
	       tokens = new String[st.countTokens()];
        }
        else
        {
        	 st = new StringTokenizer(line, separator);
        	 tokens = new String[st.countTokens()];
        }
        Activation po = null;
        int lineIndex=0;
        rating_scale.clear();
        userSet.clear();
        itemSet.clear();
        while (line != null) {
        	
        	lineIndex++;
            try{
	        	st = new StringTokenizer(line,separator);
	            
	        	int index = 0;
	         
	            
	            while (st.hasMoreTokens()) {
	                tokens[index] = st.nextToken();
	                index++;
	            }
	            int userId = Integer.parseInt(tokens[index_of_user_id].trim());
	            int itemId = Integer.parseInt(tokens[index_of_item_id].trim());
	            long timeStamp = 0L;
	            double rating;
	            if (index_of_timestamp >= 0) {
	                if (timeFormat == TimeFormat.Date) {
	                    timeStamp = dfm.parse(tokens[index_of_timestamp].trim()).getTime() / 1000;
	                } else
	                    timeStamp = Long.parseLong(tokens[index_of_timestamp]);
	                if (timeFormat == TimeFormat.Millis) {
	                    timeStamp /= 1000;
	                }
	            }
	            if(index_of_rating>=0){
	            	rating=Double.parseDouble(tokens[index_of_rating].trim());
	            	po=new RatingObservation(userId, itemId,timeStamp ,rating);
	            	rating_scale.add(rating);
	            }
	            else
	            	po = new Activation(userId, itemId, timeStamp);
	            
	            if (po != null) {
	                actions.add(po);
	                userSet.add(userId);
	                itemSet.add(itemId);
	            }
        	}catch(Exception e){
        		System.out.println("Error in processing the line "+lineIndex);
        		System.out.println(line);
        		System.out.println("Tokens");
        		for(int i=0;i<tokens.length;i++)
        			System.out.print(tokens[i]+"\t");
        		System.out.println();
        		e.printStackTrace();
        		return;
        	}
            line = br.readLine();
        }
        nUsers = userSet.size();
        nItems = itemSet.size();
        System.out.println("Dataset loaded!");
        System.out.println("------------------------------------------------");
        buildIndex();
    }

    private void buildIndex() {
        System.out.println("------------------------------------------------");
        System.out.print("Indexing data...");
        userId2Index = new Int2IntOpenHashMap();
        userIndex = new IntArrayList[nUsers];
        int u = 0;
        for (int userId : userSet) {
            userId2Index.put(userId, u);
            userIndex[u] = new IntArrayList();
            u++;
        }
        itemId2Index = new Int2IntOpenHashMap();
        itemIndex = new IntArrayList[nItems];
        int i = 0;
        for (int itemId : itemSet) {
            itemId2Index.put(itemId, i);
            itemIndex[i] = new IntArrayList();
            i++;
        }
        directIndex = new HashMap<UserItemPair, Integer>();
        Activation a;
        for (int index = 0; index < actions.size(); index++) {
            a = actions.get(index);
            directIndex.put(new UserItemPair(a.userId, a.itemId), index + 1);
            u = userId2Index.get(a.userId);
            userIndex[u].add(index + 1);
            i = itemId2Index.get(a.itemId);
            itemIndex[i].add(index + 1);
        }
        System.out.println("done!");
        System.out.println("------------------------------------------------");
    }

    public void printInfo() {
        Integer[] users = userSet.toArray(new Integer[nUsers]);
        Integer[] items = itemSet.toArray(new Integer[nItems]);
        double avgActionsPerUser = 0;
        int minActionsPerUser = Integer.MAX_VALUE;
        int maxActionsPerUser = Integer.MIN_VALUE;
        for (int u = 0; u < nUsers; u++) {
            int nActionsPerUser = getNActionsForUser(users[u]);
            avgActionsPerUser += nActionsPerUser;
            if (minActionsPerUser > nActionsPerUser)
                minActionsPerUser = nActionsPerUser;
            if (maxActionsPerUser < nActionsPerUser)
                maxActionsPerUser = nActionsPerUser;
        }
        avgActionsPerUser /= nUsers;
        double avgActionsPerItem = 0.0;
        int minActionsPerItem = Integer.MAX_VALUE;
        int maxActionsPerItem = Integer.MIN_VALUE;
        for (int i = 0; i < nItems; i++) {
            int nActionsPerItem = getNActionsForItem(items[i]);
            avgActionsPerItem += nActionsPerItem;
            if (nActionsPerItem < minActionsPerItem)
                minActionsPerItem = nActionsPerItem;
            if (nActionsPerItem > maxActionsPerItem)
                maxActionsPerItem = nActionsPerItem;
        }
        avgActionsPerItem /= nItems;
        double avgTimeBetween2UserActions = 0D;
        ArrayList<Activation> obsForUser;
        int count = 0;
        for (int u = 0; u < nUsers; u++) {
            obsForUser = getActionsForUser(users[u]);
            Collections.sort(obsForUser);
            if (obsForUser.size() > 2) {
                long min = ((Activation) obsForUser.get(0)).timeStamp;
                long max = ((Activation) obsForUser.get(obsForUser.size() - 1)).timeStamp;
                avgTimeBetween2UserActions += (max - min) / (obsForUser.size());
                count++;
            }
        }
        avgTimeBetween2UserActions /= count;
        double avgTimeBetween2ItemsUsage = 0D;
        ArrayList<Activation> obsForItem;
        count = 0;
        Integer[] itemSet = getItemSet().toArray(new Integer[nItems]);
        for (int i = 0; i < nItems; i++) {
            obsForItem = getActionsForItem(itemSet[i]);
            Collections.sort(obsForItem);
            if (obsForItem.size() > 2) {
                long min = ((Activation) obsForItem.get(0)).timeStamp;
                long max = ((Activation) obsForItem.get(obsForItem.size() - 1)).timeStamp;
                avgTimeBetween2ItemsUsage += (max - min) / (obsForItem.size());
                count++;
            }
        }
        avgTimeBetween2ItemsUsage /= count;

        double density = (double) actions.size()
                / ((long) nItems * (long) nUsers);
        double sparsity = (1 - density);
        System.out.println("------------------------------------------------");
        System.out.println(". . . . . . . . Dataset Set info . . . . . . . .");
        System.out.println("#users  \t" + nUsers);
        System.out.println("#items  \t" + nItems);
        System.out.println("#actions\t" + actions.size());
        System.out.println("density \t" + density);
        System.out.println("sparsity\t" + sparsity);
        System.out.println("Avg #actions user \t" + avgActionsPerUser);
        System.out.println("Avg #actions item \t" + avgActionsPerItem);
        System.out.println("Min #actions user \t" + minActionsPerUser);
        System.out.println("Max #actions user \t" + maxActionsPerUser);
        System.out.println("Min #actions item \t" + minActionsPerItem);
        System.out.println("Max #actions item \t" + maxActionsPerItem);
        System.out.println("Avg time 2 users actions\t"
                + avgTimeBetween2UserActions);
        System.out.println("Avg time 2 items usages \t"
                + avgTimeBetween2ItemsUsage);
        System.out.println("N Stars\t"+rating_scale.size());
        System.out.println("------------------------------------------------");
    }

    
    public int getNStars(){
    	return rating_scale.size();
    }
    
    public double[] getRatingScale(){
    	double ris[]=new double[rating_scale.size()];
    	int i=0;
    	Iterator<Double>it=rating_scale.iterator();
    	while(it.hasNext()){
    		ris[i]=it.next();
    		i++;
    	}
    	return ris;
    }
    
    public ArrayList<Activation> getActions() {
        return actions;
    }

    public IntArrayList getItemsForUser(int userId) {
        IntArrayList ris = new IntArrayList();
        int u = userId2Index.get(userId);
        IntArrayList indeces = userIndex[u];
        for (int index : indeces) {
            ris.add(actions.get(index - 1).itemId);
        }
        return ris;
    }

    public IntArrayList getUsersForItem(int itemId) {
        IntArrayList ris = new IntArrayList();
        int i = itemId2Index.get(itemId);
        IntArrayList indeces = itemIndex[i];
        for (int index : indeces) {
            ris.add(actions.get(index - 1).userId);
        }
        return ris;
    }

    public ArrayList<Activation> getActionsForUser(int userId) {
        ArrayList<Activation> ris = new ArrayList<Activation>();
        int u = userId2Index.get(userId);
        IntArrayList indeces = userIndex[u];
        for (int index : indeces) {
            ris.add(actions.get(index - 1));
        }
        return ris;
    }

    public Int2IntOpenHashMap getUserIndex() {
        return userId2Index;
    } 
    
    public ArrayList<Activation> getActionsForItem(int itemId) {
        ArrayList<Activation> ris = new ArrayList<Activation>();
        int i = itemId2Index.get(itemId);
        IntArrayList indeces = itemIndex[i];
        for (int index : indeces) {
            ris.add(actions.get(index - 1));
        }
        return ris;
    }

    public int getNActionsForItem(int itemId) {
        int i = itemId2Index.get(itemId);
        IntArrayList indeces = itemIndex[i];
        return indeces.size();
    }

    public int getNActionsForUser(int userId) {    	
    	Integer u = userId2Index.get(userId);
    	if(u==null)
    		return 0;
        IntArrayList indeces = userIndex[u];
        return indeces.size();
    }

    public TreeSet<Integer> getItemSet() {
        return itemSet;
    }

    public int getNItems() {
        return itemSet.size();
    }

    public int[] getItemArray() {
        int[] a = new int[itemSet.size()];
        int i = 0;
        for (Integer val : itemSet)
            a[i++] = val;
        return a;
    }

    public Iterator<Activation> iterator() {
        return actions.iterator();
    }

    public String getHeader() {
        return header;
    }

    public boolean existsAction(int userId, int itemId) {
        Integer dataIndex = directIndex.get(new UserItemPair(userId, itemId));
        if (dataIndex != null && dataIndex > 0) {
            return actions.get(dataIndex - 1) != null;
        } else
            return false;

    }

    public String formatAction(Activation a) {
        String s = "";
        if (index_of_user_id == 0)
            s += a.userId + separator;
        else if (index_of_item_id == 0)
            s += a.itemId + separator;
        else if (index_of_timestamp == 0)
            s += a.timeStamp + separator;
        if (index_of_user_id == 1)
            s += a.userId + separator;
        else if (index_of_item_id == 1)
            s += a.itemId + separator;
        else if (index_of_timestamp == 1)
            s += a.timeStamp + separator;

        if (index_of_user_id == 2)
            s += a.userId + separator;
        else if (index_of_item_id == 2)
            s += a.itemId + separator;
        else if (index_of_timestamp == 2)
            s += a.timeStamp + separator;

        if (index_of_user_id == 3)
            s += a.userId;
        else if (index_of_item_id == 3)
            s += a.itemId;
        else if (index_of_timestamp == 3)
            s += a.timeStamp;
        return s;
    }

    public TreeSet<Integer> getUserSet() {
        return userSet;
    }

    public int getNUsers() {
        return nUsers;
    }

    public Int2IntOpenHashMap getItemIndex() {
        return itemId2Index;
    }

    public int getSize() {
        return actions.size();
    }

    public Activation getAction(int userId, int itemId) {
        Integer dataIndex = directIndex.get(new UserItemPair(userId, itemId));
        if (dataIndex != null && dataIndex > 0) {
            return actions.get(dataIndex - 1);
        } else
            return null;
    }

    public String getSeparator() {
        return separator;
    }

   /**
    public boolean containsUser(int u) {
        return false;
    }
    **/
    
    
    public static void main(String[] args) throws Exception{
	
    	//String dataset="lastFm";
    	//String actionLogPath="resources/datasets/"+dataset+"/actionlog";
		//String readConf="resources/datasets/"+dataset+"/readConf.inf";

    //	args=new String[]{"/Users/barbieri/Documents/datasets/Flickr 2008/sample/kdd2014/feature_no_duplicates.txt",
    //	"/Users/barbieri/Documents/datasets/Flickr 2008/sample/kdd2014/readConf.inf"};

    	
    	
    	String actionLogPath="/Users/barbieri/Documents/datasets/meme2010/normalized/actionlog";
    	
		ActivationsHandler ah=new ActivationsHandler();
		ah.index_of_user_id=0;
		ah.index_of_item_id=1;
		ah.index_of_timestamp=2;
		ah.index_of_rating=-1;
		ah.skip_first_line=true;
		ah.read(actionLogPath);
		//ah.buildIndex();
		ah.printInfo();
		
	}
    
    
    
    
    
    
}