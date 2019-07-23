package evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
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
				
			//	maryam comment: 
			//double pxy=(x.size()/N )+eps;
				double pxy=(x.size()/N );
				if(pxy==0.0||Double.isInfinite(pxy)){
					//System.out.println(p_x);
					//System.out.println(p_y);
					//	maryam comment: 
					continue;
				//	throw new RuntimeException();
					
				}
				/*double help=pxy*Math.log(pxy/( p_x*p_y  ));
				if (help<0)
				{
					System.out.println(p_x+"\t"+p_y+"\t"+pxy+"\t"+help);
				}*/
				I+=pxy*Math.log(pxy/( p_x*p_y  ));
			}//for j
		}//for i
		return I;
	}//evaluateMutualInformation
	
	public static double SimilarityCom(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
	
		double N=0;
		double result=0.00;
		
		for(int i=0;i<c1.length;i++)
			N+=c1[i].size();
		
			
		for(int i=0;i<c1.length;i++){
			double check=0.00;
			int index=-1;
			for(int j=0;j<c2.length;j++){
				
				ArrayList<Integer> x=new ArrayList<Integer>(c1[i]);
				ArrayList<Integer> y=new ArrayList<Integer>(c2[j]);
				
				int x_size=x.size();
				int y_size=y.size();
				
				
		
			x.retainAll(y);
			int eshterak=x.size();
			double Oij= (eshterak)/(x_size+y_size-eshterak);
			
			if( Oij > check)
			{
				check=Oij;
				index=j;	
			}
			}
			result+=check;
		}
				
			
		return (result/N);
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

		nmi=(I/((H_x+H_y)/2));
		//nmi=I/Math.sqrt(H_x*H_y);
		//nmi=(H_x+H_y-I)/((H_x+H_y)/2);
		return nmi;
	}
	
		public static double fairNormalizeMutualInformation(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){

		//c2 is the ground (Real) C1 is the predicted: S
		int R=c2.length;
		int S=c1.length;

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
		double exponent= R-S;
		if(exponent< 0) exponent=exponent*-1;

	    	nmi=(I/((H_x+H_y)/2))*Math.exp(-1*exponent/R);
		//nmi=(I/((H_x+H_y)/2));
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
	
	
	public static void maryam_evaluate(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2, String name,String result) throws IOException{
		FileWriter Result = new FileWriter(result,true);
		
		Result.write("Name\tsimilarity\tMutual Information\tNMI\tAri\tFNMI\n");
//		System.out.println(" - - - Mutual Info Evaluation - - - - ");
//		System.out.println("Mutual Information\t\t"+mutualInformation(c1, c2));
//		System.out.println("Normalized Mutual Information\t\t"+normalizeMutualInformation(c1, c2));
//		System.out.println("Ari\t\t"+ari(c1,c2));
//		System.out.println("- - - - - - - -- -  - --  - - - - - - -");
		Result.write(name+"\t"+SimilarityCom(c1,c2)+"\t"+mutualInformation(c1, c2)+"\t"+normalizeMutualInformation(c1, c2)+"\t"+ari(c1,c2)+"\t"+fairNormalizeMutualInformation(c1,c2)+"\n");
	    Result.close();
	}
	
	
	
	public static void evaluate(ArrayList<Integer>[]c1, ArrayList<Integer>[]c2){
		
		
		
		System.out.println(" - - - Mutual Info Evaluation - - - - ");
		System.out.println("Mutual Information\t\t"+mutualInformation(c1, c2));
		System.out.println("Normalized Mutual Information\t\t"+normalizeMutualInformation(c1, c2));
		System.out.println("Fair Normalized Mutual Information\t\t"+fairNormalizeMutualInformation(c1, c2));
		System.out.println("Ari\t\t"+ari(c1,c2));
		System.out.println("- - - - - - - -- -  - --  - - - - - - -");

	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
	//	args=new String[]{"resources/one.dat","resources/two.dat"};
		String file1="";
		String file2="";
		String name="";
		String result="";
		
		for (int i = 0; i < args.length; i++) {
			
			if (args[i].equals("-c")) {
				file1 = args[i + 1];
				i++;
			}
			if (args[i].equals("-g")) {
				file2 = args[i + 1];
				i++;
			}
            if (args[i].equals("-i")) {
				
				name=args[i+1];
				i++;
		
				
			}
           if (args[i].equals("-o")) {
				
				result=args[i+1];
				i++;
		
				
			}


		}
		
		
		
		
		ArrayList<Integer>[]c1=maryamreadCommunities(file1);
		ArrayList<Integer>[]c2=maryamreadCommunities(file2);
		maryam_evaluate(c1, c2,name,result);
	}
	

	
	public static ArrayList<Integer>[] maryamreadCommunities(String file)throws Exception{
		BufferedReader br=new BufferedReader(new FileReader(file));
		int lineCount=0;
		String line=br.readLine();
		while(line!=null){
			lineCount++;
			line=br.readLine();
		}
br.close();
		@SuppressWarnings("unchecked")
		ArrayList<Integer>[] ris= new ArrayList[lineCount];
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

