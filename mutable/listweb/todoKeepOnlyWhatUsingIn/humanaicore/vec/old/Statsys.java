/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.old;
import java.util.Collection;
import java.util.Random;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.ForkMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.BitVsScalar;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.LongSize;

/** statistical system */
public interface Statsys extends LongSize, ForkMutable<Statsys>, Normable<Statsys>, BitVsScalar{
	
	/** If mutable(), returns this same Statsys. Else a Statsys modified by having learned those. */
	public Statsys think(Collection<FuzzyVec> io);
	
	/** Same as think(Collection<Statsys>) */
	public Statsys think(FuzzyVec... io);
	
	/** Returns a backed view of this Statsys which has that learnRate */
	public Statsys learnRate(double learnRate);
	
	/** Returns a backed view of this Statsys which predicts in any mutable Vec */
	public Statsys predict(boolean predict);
	
	/** Returns a backed view of this Statsys which uses that Random */
	public Statsys rand(Random rand);
	
	/** Examples: True if bayes. False if neuralnet. */
	public boolean chancesAreSymmetric();
	
	/** True if all funcs can be called by any number of threads at once.
	Example: false if is made of WeightsNodes since they store node states.
	*/
	public boolean isThreadable();
	
	/** True if is a wrapper for a simulated circuit like the bits of plus, multiply,
	a large memory thats readable andOr writable, or anything other than AI.
	Can use float64 as ptr or bit. Can even use 8 float64s as ptr at acycPair,
	(TODO) but probably Zing is better for that (which is obsoleting this and related interfaces).
	*/
	public boolean isCodeWrapper();
	
	/** Array is boolean[], double[], or LearnVec[].
	<br><br> 
	If learnRate is nonzero, learns the data in array.
	<br><br>
	If predict and learnRate is 0, simply predicts.
	If predict and learnRate is nonzero, predicts before learning,
	since if it was after learning it would be likely to return nearly the same as input.
	If !predict and learnRate==0, does nothing.
	TODO If predict and array instanceof LearnVec[]...
	<br><br>
	Consumes average of 2 bits from Random for each weighted coin flip, so SecureRandom is practical. 
	*
	public void think(Object array, double learnRate, boolean predict, Random rand);
	*/
	
	//public void think(double learnRate, boolean predict, Random rand, LearnVec... io);

}