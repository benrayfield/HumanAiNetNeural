package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.MemoryPer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZingUtil;

/** scalar vector */
public abstract class ConstSizScaVec implements MutZing{
	
	protected final double data[];
	
	/** backed array */
	public ConstSizScaVec(double[] data){
		this.data = data;
	}
	
	public long size(){ return data.length; }

	public void size(long newSize){ throw new UnsupportedOperationException(); }

	public boolean canResize(){ return false; }

	public boolean canCopy(){ return true; }

	public MutZing copyImmutableExceptAppend(){
		throw new Todo();
	}

	public long memory(){
		return MemoryPer.object*2 + MemoryPer.pointer + data.length*64;
	}

	public boolean immutableExceptAppend(){ return immutable(); }

	public boolean mutableAll(){ return !immutable(); }

	public boolean statelessData(){ return immutable(); }

	public boolean threadableData(){ return true; }

	public boolean preferScalarInsteadBit(){ return true; }

	public boolean z(long bitIndex){
		return .5<=data[(int)bitIndex];
	}

	public double d(long bitIndex){
		return data[(int)bitIndex];
	}
	
	public float f(long bitIndex){
		return (float)data[(int)bitIndex];
	}

	public long j(long bitIndex){
		return MutZingUtil.readLongAsBits(this, bitIndex);
	}

	public long ja(int longIndex){
		return j((long)longIndex<<6);
	}
	
	public int i(long bitIndex) {
		throw new Todo();
	}
	
	public int ia(int intIndex) {
		throw new Todo();
	}

	public void z(long bitIndex, boolean z){
		data[(int)bitIndex] = z ? 1 : 0;
	}

	public void d(long bitIndex, double d){
		data[(int)bitIndex] = d;
	}
	
	public void f(long bitIndex, float f){
		data[(int)bitIndex] = f;
	}

	public void j(long bitIndex, long j){
		MutZingUtil.writeLongAsBits(this, bitIndex, j);
	}
	
	public void ja(int longIndex, long j){
		j((long)longIndex<<6, j);
	}

	/*public boolean allSameValue(){
		for(int i=1; i<data.length; i++){
			if(data[0] != data[i]) return false;
		}
		return true;
	}*/
	
	public MutZing trySelfReplace(){ return this; }
	
	public boolean precisionFloat64(){ return true; }
	
	//public boolean preferFractionInsteadInverseSigmoid(){ return true; }

}