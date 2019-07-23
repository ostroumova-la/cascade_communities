
package evaluation;

import java.io.*;
import java.util.*;
import utils.MatrixUtilities;



public class EvaluateClusteringMetis
{

    public EvaluateClusteringMetis()
    {
    }

    public static void main(String args[])
        throws Exception
    {
        System.out.println(" - - - - Clustering Evaluation For Metis - - - - ");
        if(args.length == 0)
        {
            printUsage();
            return;
        }
        int nclusters = 0;
        String model = "";
        String groundTruth = "";
        for(int i = 0; i < args.length; i++)
        {
            if(args[i].equalsIgnoreCase("--help"))
            {
                printUsage();
                return;
            }
            if(args[i].equals("-c"))
            {
                nclusters = Integer.parseInt(args[i + 1]);
                i++;
            }
            if(args[i].equals("-m"))
            {
                model = args[i + 1];
                i++;
            }
            if(args[i].equals("-g"))
            {
                groundTruth = args[i + 1];
                i++;
            }
        }

        double a[][] = new double[nclusters][nclusters];
        BufferedReader in_ground = new BufferedReader(new FileReader(groundTruth));
        System.out.println((new StringBuilder("Reading ground truth from \t")).append(groundTruth).toString());
        BufferedReader in_model = new BufferedReader(new FileReader(model));
        System.out.println((new StringBuilder("Reading model from \t")).append(groundTruth).toString());
        ArrayList communities_truth[] = new ArrayList[nclusters];
        ArrayList communities_model[] = new ArrayList[nclusters];
        for(int c = 0; c < nclusters; c++)
        {
            communities_truth[c] = new ArrayList();
            communities_model[c] = new ArrayList();
        }

        HashMap t = new HashMap();
        int elem = -1;
        for(String line = in_ground.readLine(); line != null; line = in_ground.readLine())
        {
            StringTokenizer tok = new StringTokenizer(line, ",\t; ");
            elem = Integer.parseInt(tok.nextToken());
            int real_class = Integer.parseInt(tok.nextToken()) - 1;
            if(!t.containsKey(Integer.valueOf(elem)))
            {
                t.put(Integer.valueOf(elem), Integer.valueOf(real_class));
                communities_truth[real_class].add(Integer.valueOf(elem));
            }
        }

        in_ground.close();
        elem = 1;
        for(String line = in_model.readLine(); line != null; line = in_model.readLine())
        {
            int pred_class = Integer.parseInt(line);
            if(t.containsKey(Integer.valueOf(elem)))
            {
                communities_model[pred_class].add(Integer.valueOf(elem));
                int real_class = ((Integer)t.get(Integer.valueOf(elem))).intValue();
                a[real_class][pred_class]++;
            }
            elem++;
        }

        in_model.close();
        System.out.println("Confusion Matrix");
        MatrixUtilities.print(a);
        ClusteringEvaluator e = new ClusteringEvaluator(a, false);
        System.out.println((new StringBuilder("F-Index:\t\t")).append(e.getFMeasure()).toString());
        System.out.println((new StringBuilder("Jaccard Statistics:\t")).append(e.getJaccMeasure()).toString());
        System.out.println((new StringBuilder("Rand Statistics:\t")).append(e.getRandMeasure()).toString());
        System.out.println((new StringBuilder("Fowlkes Statistics:\t")).append(e.getFowlMeasure()).toString());
        System.out.println((new StringBuilder("Gamma Statistics:\t")).append(e.getGammaMeasure()).toString());
        System.out.println((new StringBuilder("Error:\t\t\t")).append(e.getError()).toString());
     //   System.out.println((new StringBuilder("Mutual:\t\t\t")).append(e.getMutualInfo()).toString());
     //   System.out.println((new StringBuilder("NMI:\t\t\t")).append(e.getNMInfo()).toString());
        System.out.println();
        System.out.println(" - - - - - - - END - - - - - - - ");
        EvaluateMutualInformation.evaluate(communities_truth, communities_model);
    }

    private static void printUsage()
    {
        System.out.println("-c <nClusters> -m <clusterModel(metisFormat)> -g <groundTruth> ");
    }
}
