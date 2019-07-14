package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.MemoryPer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZingUtil;

/** As usual in zing, size is in bits, not in units of array indexs */
public class ConstBitstring implements MutZing{
	
	public static final ConstBitstring EMPTY = new ConstBitstring(new long[0],0);
	
	public final long size;
	protected final long[] data;
	
	/** copies the array */
	public ConstBitstring(byte[] data, long size){
		throw new Todo();
	}
	
	/** copies the array */
	public ConstBitstring(byte[] data){
		this(data, data.length*8L);
	}
	
	/** copies the array */
	public ConstBitstring(long[] data, long size){
		this.data = data.clone();
		this.size = size;
	}
	
	/** copies the array */
	public ConstBitstring(long[] data){
		this(data, data.length<<6);
	}
	
	public ConstBitstring(boolean... data){
		throw new Todo();
	}
	
	public long size(){ return size; }

	public boolean canResize(){ return false; }

	public boolean canCopy(){ return true; }
	
	public MutZing copy(){ return this; }

	public MutZing copyImmutableExceptAppend(){ return this; }
	
	public MutZing mutable(boolean mutable){
		throw new Todo();
	}

	public long memory(){
		throw new Todo();
	}
	
	public boolean immutable(){ return true; }

	public boolean immutableExceptAppend(){ return true; }

	public boolean mutableAll(){ return false; }

	public boolean statelessData(){ return true; }

	public boolean threadableData(){ return true; }

	public boolean preferScalarInsteadBit(){ return false; }

	public boolean z(long bitIndex){
		throw new Todo();
	}

	public double d(long bitIndex){
		return z(bitIndex) ? 1 : 0;
	}
	
	public float f(long bitIndex){
		return z(bitIndex) ? 1 : 0;
	}

	public long j(long bitIndex){
		throw new Todo();
	}

	public long ja(int longIndex){
		throw new Todo();
	}
	
	public int i(long bitIndex){
		throw new Todo();
	}
	
	public int ia(int intIndex){
		long g = data[intIndex>>1];
		return (intIndex&1)==0 ? (int)(g>>>32) : (int)g;
	}

	public void z(long bitIndex, boolean z){
		throw new Todo();
	}

	public void d(long bitIndex, double d){
		z(bitIndex, 0<d);
	}
	
	public void f(long bitIndex, float f){
		z(bitIndex, 0<f);
	}

	public void j(long bitIndex, long j){
		throw new Todo();
	}
	
	public void ja(int longIndex, long j){
		throw new Todo();
	}

	/*public boolean allSameValue(){
		throw new Todo("cache this in constructor");
	}*/

	public void size(long newSize){
		throw new Todo();
	}
	
	public MutZing trySelfReplace(){ return this; }

	public boolean precisionFloat64(){ return false; }

	//public boolean preferFractionInsteadInverseSigmoid(){ return true; }

}