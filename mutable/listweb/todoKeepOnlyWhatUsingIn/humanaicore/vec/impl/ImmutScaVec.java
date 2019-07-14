package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.MemoryPer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZingUtil;

/** constant size immutable scalar vector */
public class ImmutScaVec extends ConstSizScaVec{
	
	protected final boolean allSameValue; //If true, should have used a different class
	
	/** Copies the array */
	public ImmutScaVec(double... numbers){
		super(numbers.clone());
		boolean same = true;
		for(int i=0; i<data.length; i++){
			double d = data[i];
			//any range is allowed if(d < 0 || 1 < d) throw new Err("Not a fraction: "+d);
			if(data[0] != d) same = false;
		}
		allSameValue = same;
	}
	
	public ImmutScaVec(MutZing z){
		super(newDoubleArray(z.size()));
		boolean same = true;
		for(int i=0; i<data.length; i++){
			data[i] = z.d(i);
			if(data[0] != data[i]) same = false;
		}
		allSameValue = same;
	}
	
	static double[] newDoubleArray(long size){
		if(Integer.MAX_VALUE < size) throw new IndexOutOfBoundsException(
			"Exceeds int range: "+size);
		return new double[(int)size];
	}
	
	public boolean allSameValue(){ return allSameValue; }

	public MutZing mutable(boolean mutable){
		//ConstSizMutScaVec not copies the array
		return mutable ? new ConstSizMutScaVec(data.clone()) : this;
	}

	public MutZing copy(){ return this; }

	public boolean immutable(){ return true; }

}