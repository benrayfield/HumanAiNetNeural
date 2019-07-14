package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ReadPrimAtBitIndex;

/** You may want to implement more funcs for efficiency, but you only have to
implement the 2 funcs to read and write long at bit index: j(long) and j(long,long).
<br><br>
For each primitive type, there are funcs for set and get at bit index,
and for each a func of long index of bit and int index of that primitive.
For example, floats are 32 bits, so the int in f(int) and fa(int,float) refers to a block of 32 bits,
and f(long) and fa(long,float) is a specific bit index that doesnt have to be on a block.
Depending on implementation, one of the funcs with int param is often most efficient,
such as f(int) and fa(int,float) wrapping a float array.
*/
public interface PrimAtBitIndex extends ReadPrimAtBitIndex, WritePrimAtBitIndex{
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default void z(long bitIndex, boolean z){
		long j = j(bitIndex);
		j(bitIndex, z ? j|0x8000000000000000L : j&0x7fffffffffffffffL);
	}
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default void b(long bitIndex, byte b){
		long j = j(bitIndex);
		j = ((b&0xffL)<<56) | (j&0x00ffffffffffffffL);
		j(bitIndex,j);
	}
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default void s(long bitIndex, short s){
		long j = j(bitIndex);
		j = ((s&0xffffL)<<48) | (j&0x0000ffffffffffffL);
		j(bitIndex,j);
	}
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */ 
	public default void i(long bitIndex, int i){
		long j = j(bitIndex);
		j = ((i&0xffffffffL)<<32) | (j&0x00000000ffffffffL);
		j(bitIndex,j);
	}
	
	

}
