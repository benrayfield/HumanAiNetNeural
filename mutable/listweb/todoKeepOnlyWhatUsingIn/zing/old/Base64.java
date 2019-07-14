package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;

public class Base64{
	private Base64(){}
	
	/** ascending order in ascii and utf8, all valid var names in java and javascript and filenames in most file systems */
	public static final String digits = "$0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
	private static final char digitChars[] = digits.toCharArray();
	private static final int charToDigit[];
	static{
		charToDigit = new int[digitChars[digitChars.length-1]+1]; //'z'+1
		for(int i=0; i<digitChars.length; i++){
			charToDigit[digitChars[i]] = i;
		}
	}
	
	/*
	"pad with bit1 then up to 5 bit0?"
	
	public static String fromBytes(byte b[]){
		return fromBytes(b, b.length*8L);
	}
	
	public static String fromBytes(byte bytes[], long bitLength){
		if(bitLength != bytes.length*8L) throw new Todo("bit alignment");
		int bits = 0;
		int bitsSize = 0;
		StringBuffer sb = new StringBuffer();
		int cycles = (int)((bitLength+7)/8);
		for(int i=0; i<cycles; i++){
			bits = (bits<<8) | (bytes[i]&0xff);
			if(i == cycles-1){ //pad with 
				
			}
			while(bits)
		}
	}
	
	public static byte[] toBytes(String base64){
		
	}
	*/
	
	/** pads with bit1 then 0-5 bit0s */
	public static String fromBits(MutZing bits){
		//int cycles = (int)((bits.size()+1+5)/6); //+1 for padding a bit1. +5 to round up
		int cycles  = (int)(bits.size()/6)+1; //+1 for padding a bit1. +5 to round up
		StringBuffer sb = new StringBuffer();
		for(int i=0; i<cycles; i++){
			int sixBits = bits.i(i*6)>>>26;
			if(i == cycles-1){ //append bit1
				int zingBitsInLastChar = (int)(bits.size()%6);
				sixBits |= 32>>zingBitsInLastChar;
			}
			sb.append(digitChars[sixBits]);
		}
		return sb.toString();
	}
	
	/** unpads from ending with bit1 then 0-5 bit0s *
	public static Zing toBits(String base64){
		int lastDigit = charToDigit[base64.charAt(base64.length()-1)];
		long bitsSize = 
	}*/

}
