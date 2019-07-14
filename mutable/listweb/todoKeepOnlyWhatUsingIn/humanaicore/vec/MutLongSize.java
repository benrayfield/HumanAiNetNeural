/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec;

public interface MutLongSize extends LongSize{
	
	/** If AskMutable.immutableExceptStreamAppend(), can only equal or increase size */
	public void size(long newSize);
	
	public boolean canResize();
	
	//"TODO merge some or all of AskMutable and ForkMutable and AskStateless and maybe LongSize"

}
