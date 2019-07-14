/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.old;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.ForkMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.BitVsScalar;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.LongSize;

/** If mutable(), setting scalar or bit sets both at once */
public interface Vec extends LongSize, BitVsScalar, ForkMutable<Vec>{

	public double scalar(long index);
	
	public boolean bit(long index);
	
	/** Useful for wrapping a bit or scalar in a Vec without paying the memory or compute cost.
	Callers expecting this behavior in some cases can call this func
	to check if the first value can be used as them all. You would want to do that
	as the confidence/weight in a TruthValue if its the same for all in another Vec.
	WARNING: In mutable Vec this may loop over them all.
	*/
	public boolean allSameValue();
	
	public void setScalar(long index, double value);
	
	public void setBit(long index, boolean value);
	
	public Vec subvec(long from, long toExclusive);
	
	public Vec setRange(long from, Vec readMe);
	
	/** backed view of this Vec as a LearnVec *
	public LearnVec learnVec(double weight);
	*/
	
	//TODO? like wavetree.Bit? public Vec cat(Vec)

}
