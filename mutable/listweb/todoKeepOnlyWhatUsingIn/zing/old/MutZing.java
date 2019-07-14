/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.AskMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.ForkMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.SelfReplacer;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.BitVsScalar;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.LongSize;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.MutLongSize;

/** WARNING: MutZing is an older version of Zing
from before float occupied 32 indexs and double occupied 64,
and thats why it still implements BitVsScalar. It shouldnt be called zing.
<br><br>
OLD TEXT...
<br><br>
Zing is a mutable or immutable or immutableExceptAppend vector or bitstring,
of mutable or immutable size, of bits or chances
which can be viewed as bit, chance, or through inverseSigmoid any range.
<br><br>
UPDATE 2016-10-12: ds funcs removed, and any range of scalars are allowed.
Converting bit to scalar gives 1 for true and 0 for false.
Converting scalar to bit gives true for positive and false for 0 or less.
See details below at "Why I removed sigmoid from Zing".
<br><br>
Example: Its efficient to store neuralWeights, which can be any range,
in a Zing if an array of those neuralWeights is literally stored
and convert with sigmoid when call funcs to get fraction/chance.
Of course, as a ForkMutable<Zing>, can get a mutable or immutable copy,
unless already immutable and ask for immutable copy it must return itself.
<br><br>
To append, use the write funcs (those with 2 params) at index size().
<br><br>
OLD BUT MOSTLY ACCURATE...
<br><br>
Zing means bitstring (TODO andOr view each index as scalar fraction).
In this case, its also AskMutable so may be mutable or immutable.
Outside bit index range 0 to size()-1, all bits are bit0, so a long can extend past a border.
<br><br>
A Zing normally wraps an array of bits (in longs) or float64s
which may be in memory, file, or Internet.
It must not compute anything, only read and write,
therefore as a ForkMutable, canCopy must always be true.
<br><br>
TODO Should there be a func that tells how expensive it is to copy,
more expensive across Internet than file than memory?
If they're all the same cost per bit, memory() compares cost accurately.
<br><br>
TODO Should Zing tell how reliable the reads and writes are,
forexample less reliable if its across Internet?
Probably not, and thats best left to minredun in acyc.
<br><br>
FIXME TODO? Like in the old Vec, each index could be both float64 andOr bit,
but I also want the ability to represent float64 as its 64 bits (or xor negs
with max long to sort them) but that can be done with a wrapper,
so Zing itself always means each bit can be float64 andOr bit.
FIXME TODO? Those float64s are always in range 0 to 1. Caller can scale that
with inverseSigmoid or linear ranges to how you want the float64s to be,
but I'm undecided if that should be part of Zing or done by caller,
such as a func that tells the min and max float64 at each index.
I think thats too much complexity in Zing since the other view is bit.
FIXME TODO? Should it be called Zing which means bitstring if it can also do scalars?
Could think of those scalars as chance since that is the normal use of them
in neuralnode states and bayesnode states and bayesweights
but is not how they're used for neural weights
or waves such as sparsedoppler or physicsmata. Caller must scale them.
<br><br>
Why I removed sigmoid from Zing:
You cant just exponent by e and take log (something like that also with division and adding) and expect to keep precision for big numbers.if its stored as fractions. I had already decided to have some zings stored as unlimited range and convert to fraction using sigmoid, but the precision of converting between them is a problem, and I need to rethink zing design. Its ok for neuralWeights, not great since if I was to convert it to fraction I'd probably want to multiply to change the stdDev. I'd hate to do it, but zing may need the complexity of a func that converts between some range and fraction or to just always use fractions? No I wont complicate zing. It has to be simple or I'll lose motivation to use it. If more complex scaling is needed, store it as fraction and let caller do that.
Maybe for pixel positions and movements and speeds scale 2^16 range into fraction range. That will work but may confuse callers.
Consider bifraction range so can forexample easily mult dy and dx without changing direction. Or how about range -.5 to .5 since its still size 1?
Or how about allow any range, and define positive as true and 0 or less as false? That works for me. So can use literal float64s again. One last thing: How to convert from bit to scalar? true 1 false 0? true 1 false -1? True .5 false -.5? I like true 1 false 0 except that the average of true and false is true. Maybe I should choose true 1 false -1. Since neuralnets and bayes uses 0 to 1, I choose to convert bit to scalar as true 1 false 0, and scalar to bit as positive is true.
*/
public interface MutZing extends MutLongSize, PrimAtBitIndex, ForkMutable<MutZing>, BitVsScalar, SelfReplacer<MutZing>{
	
	/** This refers to scalars. Bits are a .5<=scalar view,
	or it may be stored as bits and in that case scalars are all 0 or 1.
	OLD TEXT COPIED FROM VEC, WHICH THIS ZING IS REPLACING...
	Useful for wrapping a bit or scalar in a Vec without paying the memory or compute cost.
	Callers expecting this behavior in some cases can call this func
	to check if the first value can be used as them all. You would want to do that
	as the confidence/weight in a TruthValue if its the same for all in another Vec.
	WARNING: In mutable Vec this may loop over them all.
	*
	public boolean allSameValue();
	*/
	
	/*TODO Remove append funcs and use existing write funcs outside of range."
	
	/** appends 1. See AskMutable.immutableExceptAppend() *
	public void append(boolean z);
	
	/** appends 1. See AskMutable.immutableExceptAppend() *
	public void append(double fraction);
	
	/** appends 64. See AskMutable.immutableExceptAppend() *
	public void append(long j);
	*/

}