/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut;

/** Even if it only changes by roundoff, its mutable,
which could happen between different implementations of ds vs d funcs
which view range of possible scalars through sigmoid vs linearly.
*/
public interface AskMutable{
	
	/** Not the same as stateless. See comment in stateless().
	All immutable are threadable.
	*/
	public boolean immutable();
	
	/** True if immutable() or if mutable but the parts up to index size()-1 are immutable
	and that remains true after resize as each appended data is frozen.
	*/
	public boolean immutableExceptAppend();
	
	/** If is a Zing forExample, means all indexs can be written,
	so in that case its the same as immutable() && immutableExceptStreamAppend().
	TODO Is this func needed? Only if theres other things than Zing this is relevant to,
	and its convenient and more efficient to only call 1 func.
	*/
	public boolean mutableAll();
	
	/** Not the same as immutable. An immutable can be stateless or not.
	Example: On stateful immutable Mind, call think(Zing)
	returns a stateful immutable Mind that has
	learned from (or changed its state in some way) that Zing.
	All mutable are stateful. Different than threadable because
	all immutable are threadable, but immutable can be stateful.
	*/
	public boolean statelessData();
	
	/** Only relevant if !immutable(). Immutable things are always threadable.
	True if all funcs can be called by any number of threads at once.
	Example: false if is made of WeightsNodes since they store node states.
	<br><br>
	FIXME: When 1 thread reads a long or double while another thread writes it,
	even if its ok for the order of those to be random, its not atomic
	so is not threadable for that op, even if its just a wrapper for an array.
	*/
	public boolean threadableData();

}
