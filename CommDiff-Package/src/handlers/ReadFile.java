package handlers;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class ReadFile {

	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws Exception {
		BufferedReader br=new BufferedReader(new FileReader(args[0]));
		PrintWriter pw=new PrintWriter(new FileWriter("ouput"));
		String line=br.readLine();
		
		
		while(line!=null){
				pw.println(line);
			line=br.readLine();
		}
		
		
		pw.close();
		pw.flush();

	}

}
