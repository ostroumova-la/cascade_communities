package core;

import it.unimi.dsi.fastutil.ints.*;

import java.io.*;
import java.util.*;

import beans.*;
import handlers.*;

public class LinksHandlerUtil {

    private Random r = new Random(31);

    private Set<Edge> trainingEdges = new HashSet<Edge>();
    private Set<Edge> testEdges = new HashSet<Edge>();

    /**
     * 
     * @param links
     * @param actions
     * @param ratio
     *            porzione del test set
     * @throws Exception
     */
    public void generateTrainingAndTestSet(LinksHandler links, double ratio,
            String trainingFile, String testFile) throws Exception {
        int expectedSize = (int) (ratio * links.getNDirectedEdges());
        System.out.println("Expected testset size = " + expectedSize + "/"
                + links.getNDirectedEdges());
        int[] ids = links.getVertexId2Index().keySet().toIntArray();
        for (int userId : ids) {
            IntArrayList friends = links.getOutLinksForVertex(userId);
            if (friends.size() > 0) {
                int percent = (int) (ratio * friends.size());
                int count = 0;
                for (int friendId : friends) {
                    if (count < percent) {
                        testEdges.add(new Edge(userId, friendId));
                    } else {
                        trainingEdges.add(new Edge(userId, friendId));
                    }
                    count++;
                }
            }
        }
        int iteration = 1;
        System.out.println("#it\t#!train\tremaining");
        System.out.println("-------------------------");
        while (true) {
            IntSet usersNotInTraining = computeUsersNotInTraining();
            int remaining = expectedSize - testEdges.size();
            System.out.println((iteration++) + "\t" + usersNotInTraining.size()
                    + "\t" + remaining);
            if (usersNotInTraining.size() == 0 && Math.abs(remaining) <= 10) {
                System.out.println("---------------------");
                System.out.println("All:\t" + links.getNDirectedEdges());
                System.out.println("Train:\t" + trainingEdges.size());
                System.out.println("Test:\t" + testEdges.size());
                break;
            } else if (Math.abs(remaining) > 10) {
                Set<Edge> randomEdges = selectRandomFromTraining(remaining);
                trainingEdges.removeAll(randomEdges);
                testEdges.addAll(randomEdges);
            } else if (usersNotInTraining.size() > 0) {
                Set<Edge> randomEdges = selectFromTest(usersNotInTraining);
                testEdges.removeAll(randomEdges);
                trainingEdges.addAll(randomEdges);
            } else {
                throw new RuntimeException("Unexpected state");
            }
        }
        PrintWriter trainPw = new PrintWriter(trainingFile);
        trainPw.println("userId\tfriendId");
        for (Edge e : trainingEdges) {
            trainPw.println(e.source + "\t" + e.destination);
        }
        trainPw.flush();
        trainPw.close();

        PrintWriter testPw = new PrintWriter(testFile);
        testPw.println("userId\tfriendId");
        for (Edge e : testEdges) {
            testPw.println(e.source + "\t" + e.destination);
        }
        testPw.flush();
        testPw.close();
    }

    public void generateKFolds(LinksHandler dataHandler, int K, String path)
            throws IOException {
        if (K <= 0)
            throw new RuntimeException("K must be greater than zero");
        // init
        PrintWriter pws[] = new PrintWriter[K];
        for (int k = 0; k < K; k++) {
            pws[k] = new PrintWriter(path + "/Fold" + k);
            pws[k].println("userId\tfriendId");
        }

        Set<Edge> edges = dataHandler.getEdges();
        for (Edge e : edges) {
            int index = (int) (Math.random() * K);
            pws[index].println(e.source + "\t" + e.destination);
        }
        for (int k = 0; k < K; k++) {
            pws[k].flush();
            pws[k].close();
        }
    }

    private IntSet computeUsersNotInTraining() {
        IntSet trainingUsers = new IntOpenHashSet();
        IntSet testUsers = new IntOpenHashSet();
        for (Edge e : trainingEdges) {
            trainingUsers.add(e.source);
            trainingUsers.add(e.destination);
        }
        for (Edge e : testEdges) {
            testUsers.add(e.source);
            testUsers.add(e.destination);
        }
        IntSet usersNotInTraining = new IntOpenHashSet();
        for (int u : testUsers) {
            if (!trainingUsers.contains(u)) {
                usersNotInTraining.add(u);
            }
        }
        return usersNotInTraining;
    }

    private Set<Edge> selectRandomFromTraining(int size) {
        Set<Edge> ris = new HashSet<Edge>();
        while (true) {
            for (Edge e : trainingEdges) {
                if (r.nextDouble() < 0.5) {
                    ris.add(e);
                }
                if (ris.size() == size) {
                    return ris;
                }
            }
        }
    }

    private Set<Edge> selectFromTest(IntSet usersNotIntTraining) {
        Set<Edge> ris = new HashSet<Edge>();
        for (int u : usersNotIntTraining) {
            Iterator<Edge> it = testEdges.iterator();
            while (it.hasNext()) {
                Edge e = it.next();
                if (e.source == u || e.destination == u) {
                    ris.add(e);
                }
                if (ris.size() == usersNotIntTraining.size()) {
                    break;
                }
            }
        }
        return ris;
    }

    public static void main(String[] args) {
        try {
            LinksHandler links = new LinksHandler();
            links.read("citations.txt");
            links.printInfo();
            LinksHandlerUtil g = new LinksHandlerUtil();
            g.generateTrainingAndTestSet(links, .3, "trainingCitations.txt",
                    "testCitations.txt");
            LinksHandler test = new LinksHandler();
            test.read("testCitations.txt");
            test.printInfo();
            g.generateKFolds(test, 3, "train");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
