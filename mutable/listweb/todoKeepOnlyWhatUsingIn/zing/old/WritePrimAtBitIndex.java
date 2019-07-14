package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

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
public interface WritePrimAtBitIndex{
	
	public void z(long bitIndex, boolean z);
	
	public void b(long bitIndex, byte b);
	public default void ba(int byteIndex, byte b){ b(((long)byteIndex)<<3,b); }
	
	public void s(long bitIndex, short s);
	public default void sa(int shortIndex, short s){ s(((long)shortIndex)<<4,s); }
	

	public void i(long bitIndex, int i);
	public default void ia(int intIndex, int i){ i(((long)intIndex)<<5,i); }
	
	/** this is 1 of the 2 funcs you must implement */
	public void j(long bitIndex, long j);
	public default void ja(int longIndex, long j){ j(((long)longIndex)<<6,j); }
	
	public default void f(long bitIndex, float f){ i(bitIndex, Float.floatToRawIntBits(f)); }
	public default void fa(int floatIndex, float f){ f(((long)floatIndex)<<5,f); }
	
	public default void d(long bitIndex, double d){ j(bitIndex, Double.doubleToRawLongBits(d)); }
	public default void da(int longIndex, double d){ d(((long)longIndex)<<6,d); }

}
