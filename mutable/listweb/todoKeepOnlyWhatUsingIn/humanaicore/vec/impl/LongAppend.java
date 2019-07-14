package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.MemoryPer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZingUtil;

/** A growable (by doubling size when needed) array of longs
that can only append but not write in existing size.
Useful for acyc64 in tlapp (torrent like acyc part packet).
*/
public class LongAppend implements MutZing{
	
	/** size is in bits as usual in Zing */
	protected long size;
	protected long[] data = new long[1];
	
	public LongAppend(){
		data = new long[1];
	}
	
	/** backed array. To be used only for internal copy */
	protected LongAppend(long[] data, long size){
		if(!MathUtil.isPowerOf2(data.length)) throw new Err(
			"Not powOf2 bits: "+data.length*64);
		this.size = size;
		this.data = data;
	}
	
	public long size(){ return size; }

	/** Fills with bit0 up to new size, doubling array size as many times as needed.
	This is normally only called internally since external caller probably wants to
	append specific data instead of all 0s, and only internal code can be trusted
	to write those only 1 time just after that since this is immutableExceptAppend.
	*/
	public synchronized void size(long newSize){
		long size = size(); //in case it changes during this
		if(newSize < size) throw new IndexOutOfBoundsException("newSize="+newSize+" < size="+size);
		while(data.length*64 < newSize){
			long[] data2 = new long[data.length*2];
			System.arraycopy(data, 0, data2, 0, (int)((size+63)/64));
		}
		size = newSize;
	}

	public boolean canResize(){ return true; }

	public boolean canCopy(){ return true; }
	
	public MutZing copy(){ return new LongAppend(data.clone(), size); }

	public MutZing copyImmutableExceptAppend(){ return copy(); }
	
	public MutZing mutable(boolean mutable){
		if(mutable) return this;
		throw new Todo("Copy to immutable bitstring Zing");
	}

	public long memory(){
		return MemoryPer.object*2 + MemoryPer.pointer + 64 + data.length*64;
	}
	
	public boolean immutable(){ return false; }

	public boolean immutableExceptAppend(){ return true; }

	public boolean mutableAll(){ return false; }

	public boolean statelessData(){ return false; }

	public boolean threadableData(){
		throw new Todo("https://www.reddit.com/r/javahelp/comments/55djq9/without_using_an_object_or_synchronized_is_there/");
	}

	public boolean preferScalarInsteadBit(){ return false; }
	
	public boolean precisionFloat64(){ return false; }

	public boolean z(long bitIndex){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}

	public double d(long bitIndex){
		return z(bitIndex) ? 1 : 0;
	}
	
	public float f(long bitIndex){
		return z(bitIndex) ? 1 : 0;
	}

	public long j(long bitIndex){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}

	public long ja(int longIndex){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}
	
	public int i(long bitIndex) {
		throw new Todo();
	}
	
	public int ia(int intIndex) {
		throw new Todo();
	}

	public void z(long bitIndex, boolean z){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}

	public void d(long bitIndex, double d){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}
	
	public void f(long bitIndex, float f){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}

	public void j(long bitIndex, long j){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}
	
	public void ja(int longIndex, long j){
		throw new Todo("read allowed. write allowed only if index is size and then increase size");
	}

	/*public boolean allSameValue(){
		if(size%64 != 0) throw new Todo("Allow append bit?");
		for(int i=0; i<data.length; i++){
			if((data[i] != 0 && data[i] != -1) || data[0] != data[i]) return false;
		}
		return true;
	}*/
	
	public MutZing trySelfReplace(){ return this; }
	
	//public boolean preferFractionInsteadInverseSigmoid(){ return true; }

}