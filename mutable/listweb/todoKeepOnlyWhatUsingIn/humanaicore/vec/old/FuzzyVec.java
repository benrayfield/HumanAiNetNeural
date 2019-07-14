/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.old;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.AskMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.LongSize;

/** As an AskMutable, returns weight().mutable() && value.mutable().
The pointers to those 2 Vec are immutable.
*/
public interface FuzzyVec extends LongSize, AskMutable{
	
	/** This is normally multiplied by learnRate so each trainingData can have its own.
	Confidence or weight in a TruthValue. Range 0 to 1.
	Immutable pointer to a mutable or immutable Vec.
	For efficiency, check weight().allSameValue() before using them separately,
	which is common in giving a single weight to each trainingData.
	*/
	public Vec weight();
	
	/** Immutable pointer to a mutable or immutable Vec. */
	public Vec value();

}
