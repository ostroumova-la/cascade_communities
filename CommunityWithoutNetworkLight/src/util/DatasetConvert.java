package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class DatasetConvert {

	/**
	 * @param args
	 */
	public static void main(String[] args)throws Exception {
		PrintWriter pw=new PrintWriter(new FileWriter("actionlog"));
		
		BufferedReader br=new BufferedReader(new FileReader(args[0]));
		String line=br.readLine();
		int itemId=0;
		line=br.readLine();
		StringTokenizer st;
		while(line!=null){	
			st=new StringTokenizer(line,"\t");
			if(st.countTokens()==2){
				itemId++;
			}
			else{
				st.nextToken();
				String user=st.nextToken();
				String time=st.nextToken(); 
				pw.println(""+user+"\t"+itemId+"\t"+time);
			}
			line=br.readLine();
		}
		
		pw.flush();
		pw.close();
		
		
		System.out.println(""+itemId+" items");
		
		

	}

}
