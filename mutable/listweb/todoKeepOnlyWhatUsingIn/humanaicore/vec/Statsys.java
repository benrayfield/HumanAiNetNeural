/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec;
import java.util.Random;

public interface Statsys<T extends Statsys<T>> extends Mind<T>{
	
	/** Returns a backed view of this Statsys which has that learnRate */
	public T learnRate(double learnRate);
	
	/** Returns a backed view of this Statsys which predicts in any mutable Vec.
	If not predict, then writes().allSameValue()==true and those values are all false,
	which means it only reads and (if learnRate is nonzero) learns from them.
	*/
	public T predict(boolean predict);
	
	/** Returns a backed view of this Statsys which uses that Random */
	public T rand(Random rand);

}
