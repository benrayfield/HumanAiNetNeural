package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.MemoryPer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZingUtil;

/** constant size mutable scalar vector */
public class ConstSizMutScaVec extends ConstSizScaVec{
	
	public ConstSizMutScaVec(int size){
		this(new double[size]);
	}
	
	/** backing array. Must always be fractions. */
	public ConstSizMutScaVec(double... fractions){
		super(fractions);
	}
	
	public MutZing mutable(boolean mutable){
		return mutable ? this : new ImmutScaVec(data); //ImmutScaVec copies the array
	}

	public MutZing copy(){
		return new ConstSizMutScaVec(data.clone());
	}

	public boolean immutable(){ return false; }

}