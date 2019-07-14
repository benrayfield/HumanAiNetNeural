/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut;

public interface ForkMutable<T> extends AskMutable{
	
	/** Its possible for a Statsys to be mutable() but not canCopy().
	Example: a Statsys which wraps the process of a person using the program.
	Copy is defined as: (mutable() && call copy()) or call mutable(!mutable())
	*/
	public boolean canCopy();
	
	/** If param not equals mutable(), copies to a new T thats mutableAll() or immutable() */
	public T mutable(boolean mutable);
	
	/** Returns a copy of this which is immutableExceptAppend() and therefore mutable() */
	public T copyImmutableExceptAppend();
	
	/** If already mutable and you want another mutable copy without first making an immutable copy.
	If immutable, returns this.
	*/
	public T copy();
	
	/** estimate of bits more memory will be used if (mutable() && call copy()) or if call mutable(!mutable()) */
	public long memory();
	
	//"TODO merge some or all of AskMutable and ForkMutable and AskStateless and maybe LongSize"

}
