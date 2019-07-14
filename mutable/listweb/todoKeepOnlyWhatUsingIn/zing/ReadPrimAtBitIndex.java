package mutable.listweb.todoKeepOnlyWhatUsingIn.zing;

/** TODO rewrite this text which was before PrimAtBitIndex was split into read and write types.
<br><br>
You may want to implement more funcs for efficiency, but you only have to
implement the 2 funcs to read and write long at bit index: j(long) and j(long,long).
<br><br>
For each primitive type, there are funcs for set and get at bit index,
and for each a func of long index of bit and int index of that primitive.
For example, floats are 32 bits, so the int in f(int) and fa(int,float) refers to a block of 32 bits,
and f(long) and fa(long,float) is a specific bit index that doesnt have to be on a block.
Depending on implementation, one of the funcs with int param is often most efficient,
such as f(int) and fa(int,float) wrapping a float array.
*/
public interface ReadPrimAtBitIndex{
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default boolean z(long bitIndex){
		return (j(bitIndex)&0x8000000000000000L) != 0;
	}
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default byte b(long bitIndex){
		return (byte)(j(bitIndex)>>>56);
	}
	public default byte ba(long byteIndex){ return b(byteIndex<<3); }
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default short s(long bitIndex){
		return (short)(j(bitIndex)>>>48);
	}
	public default short sa(long shortIndex){ return s(shortIndex<<4); }
	
	/** This could be much more efficient if coded for a specific kind of array in an implementing class */
	public default int i(long bitIndex){
		return (int)(j(bitIndex)>>>32);
	}
	public default int ia(long intIndex){ return i(intIndex<<5); }
	
	/** implement this */
	public long j(long bitIndex);
	public default long ja(long longIndex){ return j(longIndex<<6); }
	
	public default float f(long bitIndex){ return Float.intBitsToFloat(i(bitIndex)); }
	public default float fa(long floatIndex){ return f(floatIndex<<5); }
	
	public default double d(long bitIndex){ return Double.longBitsToDouble(j(bitIndex)); }
	public default double da(long doubleIndex){ return d(doubleIndex<<6); }

}