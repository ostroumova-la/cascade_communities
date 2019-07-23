package core;

import it.unimi.dsi.fastutil.ints.*;

import java.io.*;
import java.util.*;


import beans.*;
import handlers.*;

public class ActionsHandlerUtils {

    private static long seed = 7919;
    private static Random random = new Random(seed);

    public static void generateTraningAndTestDataset(
            ActivationsHandler dataHandler, double ratio, String trainingPath,
            String testPath) throws IOException {
        ActivationsHandler test = generateHeldOutDataset(dataHandler, ratio);
        Iterator<Activation> it = dataHandler.iterator();
        PrintWriter pwTraining = new PrintWriter(new FileWriter(trainingPath));
        PrintWriter pwTest = new PrintWriter(new FileWriter(testPath));
        pwTraining.println(dataHandler.getHeader());
        pwTest.println(dataHandler.getHeader());
        while (it.hasNext()) {
            Activation a = it.next();
            if (test.existsAction(a.userId, a.itemId)) {
                pwTest.print(dataHandler.formatAction(a) + "\n");
            } else {
                pwTraining.print(dataHandler.formatAction(a) + "\n");
            }
        }
        pwTraining.flush();
        pwTraining.close();
        pwTest.flush();
        pwTest.close();
    }

    public static ActivationsHandler generateHeldOutDataset(
            ActivationsHandler dataHandler, double heldOutRatio) {
        System.out.println("Generating held out dataset");
        if (heldOutRatio <= 0)
            return new ActivationsHandler();
        int n_users = dataHandler.getNUsers();
        Integer[] users = dataHandler.getUserSet()
                .toArray(new Integer[n_users]);
        int currentSize = 0;
        int expectedSize = (int) (dataHandler.getSize() * heldOutRatio);
        TreeSet<Integer> testUsers = new TreeSet<Integer>();
        TreeSet<Integer> testItems = new TreeSet<Integer>();
        Set<UserItemPair> testPairs = new HashSet<UserItemPair>();
        HashMap<Integer, IntArrayList> user2Items = new HashMap<Integer, IntArrayList>();
        System.out.println("size: " + expectedSize + "/"
                + dataHandler.getSize());
        while (currentSize < expectedSize) {
            int randomId = users[(int) (random.nextDouble() * n_users)];
            ArrayList<Activation> actionsForRandomUser = dataHandler
                    .getActionsForUser(randomId);
            if (actionsForRandomUser.size() > 0) {
                Collections.sort(actionsForRandomUser);
                for (Activation a : actionsForRandomUser) {
                    UserItemPair pair = new UserItemPair(a.userId, a.itemId);
                    if (!testPairs.contains(pair)) {
                        testUsers.add(a.userId);
                        testItems.add(a.itemId);
                        testPairs.add(pair);
                        currentSize++;
                        IntArrayList userItemList = user2Items.get(a.userId);
                        if (userItemList == null) {
                            userItemList = new IntArrayList();
                            user2Items.put(a.userId, userItemList);
                        }
                        userItemList.add(a.itemId);
                        if (currentSize >= expectedSize) {
                            break;
                        }
                    }
                }
            }
        }
        ArrayList<Activation> testData = new ArrayList<Activation>();
        for (int user : testUsers) {
            IntArrayList userItemList = user2Items.get(user);
            for (int item : userItemList) {
                Activation a = dataHandler.getAction(user, item);
                if (a == null) {
                    System.out.println(user);
                    System.out.println(item);
                    IntArrayList ulist = dataHandler.getItemsForUser(user);
                    if (ulist.contains(item)) {
                        System.out.println("ok");
                    }
                    throw new RuntimeException();
                }
                testData.add(a.getCopy());
            }
        }
        System.out.println("Generating held out dataset: DONE");
        ActivationsHandler ris = new ActivationsHandler(testUsers, testItems, testData);
        return ris;
    }

    public static boolean checkTrainingAndTestSetConsistent(
            ActivationsHandler training, ActivationsHandler test) {
        boolean consistent = true;
        IntSet usersNotInTraining = new IntOpenHashSet();
        IntSet itemsNotInTraining = new IntOpenHashSet();
        ArrayList<Activation> duplicateAction = new ArrayList<Activation>();
        for (int u : test.getUserSet()) {
            if (training.getNActionsForUser(u) == 0) {
                consistent = false;
                usersNotInTraining.add(u);
            }
        }
        for (int itemTestSet : test.getItemSet()) {
            if (training.getNActionsForItem(itemTestSet) == 0) {
                consistent = false;
                itemsNotInTraining.add(itemTestSet);
            }
        }
        for (Activation a : test.getActions()) {
            if (training.existsAction(a.userId, a.itemId)) {
                duplicateAction.add(a);
                consistent = false;
            }
        }
        ArrayList<Activation> list;
        if (!consistent) {
            System.out.println("The following users are in the test set "
                    + "but do not belong to the training set");
            for (int userId : usersNotInTraining) {
                System.out.println("<" + userId + ">");
                list = test.getActionsForUser(userId);
                for (Activation ro1 : list) {
                    System.out.println("\t" + ro1);
                }
                System.out.println("----------------------------");
            }
            System.out.println();
            System.out
                    .println("The following items are in the test set but do not belong to the training set");
            for (int itemId : itemsNotInTraining) {
                System.out.println("<" + itemId + ">");
                list = test.getActionsForItem(itemId);
                for (Activation ro1 : list) {
                    System.out.println("\t" + ro1);
                }
                System.out.println("----------------------------");
            }
            System.out.println();
            if (duplicateAction.size() > 0) {
                System.out
                        .println("The following rating observation are duplicates");
                for (Activation ro1 : duplicateAction)
                    System.out.println(ro1);
            }
        }
        return consistent;
    }

    public static void generateKFoldSamples(ActivationsHandler dataHandler, int K)
            throws Exception {
        if (K <= 0)
            throw new RuntimeException("K must be greater than zero");
        int n_users = dataHandler.getNUsers();
        Integer[] users = dataHandler.getUserSet()
                .toArray(new Integer[n_users]);
        // init
        PrintWriter pws[] = new PrintWriter[K];
        for (int k = 0; k < K; k++) {
            pws[k] = new PrintWriter(new FileWriter("Sample" + k));
            pws[k].println(dataHandler.getHeader());
        }
        for (int u = 0; u < n_users; u++) {
            ArrayList<Activation> actionsForUser = dataHandler
                    .getActionsForUser(users[u]);
            for (int i = 0; i < actionsForUser.size(); i++) {
                Activation a = actionsForUser.get(i);
                int index = (int) (Math.random() * K);
                pws[index].println(a.userId + "\t" + a.itemId + "\t"
                        + a.timeStamp);
            }
        }
        for (int k = 0; k < K; k++) {
            pws[k].flush();
            pws[k].close();
        }
    }
    
    
    public static void exportSVDLIBC_Dense(ActivationsHandler data, String file)throws Exception{
    	PrintWriter pw=new PrintWriter(new FileWriter(file,true));
    	
   
    	    
    	int numRows=data.getNUsers();
    	int numCols=data.getNItems();
    	
    	System.out.println("numrows\t"+numRows);
    	
    	System.out.println("numcols\t"+numCols);

        Integer[] users = data.getUserSet().toArray(new Integer[numRows]);

        Integer[] items = data.getItemSet().toArray(new Integer[numCols]);

    	pw.println(""+numRows+" "+numCols);
    	
    	System.out.println(numRows);
    	System.out.println(numCols);
    	
    	for(int u=0;u<numRows;u++){
    		for(int i=0;i<numCols;i++){
    			double r=0.0;
    			RatingObservation ro=(RatingObservation)data.getAction(users[u], items[i]);
    			if(ro!=null)
    				r=ro.getRating();
    			pw.print(""+r+" ");
    		}
    		
    		pw.print("\n");
    	}
    	
    	pw.flush();
    	pw.close();
    }

	
    public static void main(String[] args) throws Exception {
        ActivationsHandler data = new ActivationsHandler();
        data.read("train/dataset/actions.txt", "train/dataset/conf.inf");
        data.printInfo();
        ActivationsHandler training = new ActivationsHandler();
        ActivationsHandler test = new ActivationsHandler();
        generateTraningAndTestDataset(data, 0.2,
                "diggActionsTest/trainingActions.txt",
                "diggActionsTest/testActions.txt");
        training.read("diggActionsTest/trainingActions.txt",
                "train/dataset/conf.inf");
        test.read("diggActionsTest/testActions.txt", "train/dataset/conf.inf");
        System.out.println(checkTrainingAndTestSetConsistent(training, test));
    }
}