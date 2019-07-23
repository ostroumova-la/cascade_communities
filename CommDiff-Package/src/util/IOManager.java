package util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.zip.GZIPOutputStream;
import java.util.zip.GZIPInputStream;


public  class IOManager {

	public static void writeObjectOnFile(Serializable s, String fileName) throws Exception {
		ObjectOutputStream out = null;
		try {
			GZIPOutputStream gz = new GZIPOutputStream(new FileOutputStream(new File(fileName)));
			out = new ObjectOutputStream(gz);
			out.writeObject(s);
			out.flush();
			gz.flush();
			gz.finish();
			out.close();
		} catch(Exception ex) {
			try {
				ex.printStackTrace();
				out.close();
			} catch(Exception ex2) {
				ex2.printStackTrace();
			}
			throw ex;
		}
	}//writeObjectOnFile
	
	
	public static Object loadObjectFromFile(String fileName)throws Exception{
		ObjectInputStream in = null;
		Object o=null;
		try {
			in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(new File(fileName))));
			o= in.readObject();
			in.close();
			return o;
		} catch(Exception ex) {
			try {
				ex.printStackTrace();
				in.close();
			} catch(Exception ex2) {
				ex2.printStackTrace();
			}
			throw ex;
		}
	}//loadObjectFromFile
	


	
	public static void writeMatrixOnFile(double[][]matrix,String filename) throws Exception{
		PrintWriter pw = new PrintWriter(new FileWriter(filename));
		
		for(int i=0;i<matrix.length;i++)
		{
			for(int j=0;j<matrix[0].length;j++)
				pw.print(""+matrix[i][j]+"\t");
		
			pw.print("\n");
		}
		
		
		pw.flush();
		pw.close();
		
	}
	
	public static void writeArrayOnFile(double[]array,String filename) throws Exception{
		PrintWriter pw = new PrintWriter(new FileWriter(filename));
		
		for(int i=0;i<array.length;i++)
		{
				pw.print(""+array[i]+"\t");
			
		}
		pw.print("\n");
		
		pw.flush();
		pw.close();
		
	}
	
	
	
}
