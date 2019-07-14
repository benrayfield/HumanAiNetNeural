/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.wavetree.bit.object;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.wavetree.bit.Bits;

public interface HeaderAndData{
	
	public Bits data();

	/** may be empty */
	public Bits header();
	
	public Bits headerThenData();

}
