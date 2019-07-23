package evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.StringTokenizer;


public class EvaluateMutualInformation {

	static double eps=Double.MIN_VALUE;
	
	/*
	 *
	def mutual_info(x,y):
    N=double(x.size)
    I=0.0
    eps = numpy.finfo(float).eps
    for l1 in unique(x):
        for l2 in unique(y):
            #Find the intersections
            l1_ids=nonzero(x==l1)[0]
            l2_ids=nonzero(y==l2)[0]
            pxy=(double(intersect1d(l1_ids,l2_ids).size)/N)+eps
            I+=pxy*log2(pxy/((l1_ids.size/N)*(l2_ids.size/N)))
    return I
	 */
	
	public static double mutualInformation(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
		double I=0.0;
		double N=0;
		for(int i=0;i<c1.length;i++)
			N+=c1[i].size();
		
			
		for(int i=0;i<c1.length;i++){
			for(int j=0;j<c2.length;j++){
				
				ArrayList<Integer> x=new ArrayList<Integer>(c1[i]);
				ArrayList<Integer> y=new ArrayList<Integer>(c2[j]);
				
				int x_size=x.size();
				int y_size=y.size();
				
				double p_x=(double)x_size/N ;
				double p_y=(double)y_size/N ;
				
				
				x.retainAll(y);
				
				double pxy=(x.size()/N )+eps;
				if(pxy==0.0||Double.isInfinite(pxy)){
					System.out.println(p_x);
					System.out.println(p_y);
					throw new RuntimeException();
				}
				I+=pxy*Math.log(pxy/( p_x*p_y  ));
			}//for j
		}//for i
		return I;
	}//evaluateMutualInformation
	
	
	/*
	 *  symmetric uncertainty (Witten & Frank 2005)
	 *   represents a weighted average of the two uncertainty coefficients (Press & Flannery 1988)
	def nmi(x,y):
    N=x.size
    I=mutual_info(x,y)
    Hx=0
    for l1 in unique(x):
        l1_count=nonzero(x==l1)[0].size
        Hx+=-(double(l1_count)/N)*log2(double(l1_count)/N)
    Hy=0
    for l2 in unique(y):
        l2_count=nonzero(y==l2)[0].size
        Hy+=-(double(l2_count)/N)*log2(double(l2_count)/N)
    return I/((Hx+Hy)/2)
	 */
	
	
	public static double normalizeMutualInformation(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
		double N=0;
		for(int i=0;i<c1.length;i++)
			N+=c1[i].size();
		
		double nmi=0;
		double I=mutualInformation(c1, c2);
		
		double H_x=0;
		for(int i=0;i<c1.length;i++){
			double p_x_i=(double)c1[i].size()/N;
			H_x+=-p_x_i*Math.log(p_x_i);	
		}
		
		double H_y=0;
		for(int j=0;j<c2.length;j++){
			double p_y_j=(double)c2[j].size()/N;
			H_y+=-p_y_j*Math.log(p_y_j);	
		}
		nmi=I/((H_x+H_y)/2);
		//nmi=I/Math.sqrt(H_x*H_y);
		//nmi=(H_x+H_y-I)/((H_x+H_y)/2);
		return nmi;
	}
	
	
	public static double ari(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
		
		double num=0;
		double den=0;
		
		double N=0;
		for(int i=0;i<c1.length;i++)
			N+=c1[i].size();
		
		
		double binomial_n_2=N*(N-1)/2;
		
		double a=0.0;
		
		for(int i=0;i<c1.length;i++){
			
			for(int j=0;j<c2.length;j++){
				
				ArrayList<Integer> x=new ArrayList<Integer>(c1[i]);
				ArrayList<Integer> y=new ArrayList<Integer>(c2[j]);
				
				
				x.retainAll(y);
				int n_ij=x.size();
				
				a+=n_ij*(n_ij-1)/2;
				
			}//for j
		}//for i
		
		double b=0;
		double c=0;
		
		for(int i=0;i<c1.length;i++){
			int n_i=c1[i].size();
			b+=n_i*(n_i-1)/2;
		}
		
		for(int j=0;j<c2.length;j++){
			int n_j=c2[j].size();
			c+=n_j*(n_j-1)/2;
		}
		
		num=a-(b*c)/binomial_n_2;
		den= 0.5*(b+c) - (b*c  )/binomial_n_2;
		
		
		return num/den;
	}
	
	
	public static void evaluate(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
		System.out.println(" - - - Mutual Info Evaluation - - - - ");
		System.out.println("Mutual Information\t\t"+mutualInformation(c1, c2));
		System.out.println("Normalized Mutual Information\t\t"+normalizeMutualInformation(c1, c2));
		System.out.println("Ari\t\t"+ari(c1,c2));
		System.out.println("- - - - - - - -- -  - --  - - - - - - -");
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		args=new String[]{"resources/one.dat","resources/two.dat"};
		String file1=args[0];
		String file2=args[1];
		ArrayList<Integer>[]c1=readCommunities(file1);
		ArrayList<Integer>[]c2=readCommunities(file2);
		evaluate(c1, c2);
	}
	
	
	public static ArrayList<Integer>[] readCommunities(String file)throws Exception{
		BufferedReader br=new BufferedReader(new FileReader(file));
		int lineCount=0;
		String line=br.readLine();
		while(line!=null){
			lineCount++;
			line=br.readLine();
		}

		ArrayList<Integer>[] ris=new ArrayList[lineCount];
		for(int c=0;c<lineCount;c++)
			ris[c]=new ArrayList<Integer>();
		
		
		br=new BufferedReader(new FileReader(file));
		line=br.readLine();
		StringTokenizer st;
		int c=0;
		while(line!=null){
			st=new StringTokenizer(line);
			while(st.hasMoreTokens()){
				ris[c].add(Integer.parseInt(st.nextToken()));
			}
			c++;
			line=br.readLine();
		}	
		return ris;
	}

}//EvaluateMutualInformation
