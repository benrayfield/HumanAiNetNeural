package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.langs;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl.ConstBitstring;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.*;

/** translates between float64 and zing of those content bits.
Most primitives would be used in a big array in a single zing,
such as neuralnet using a zing of many float32,
instead of this inefficient translation,
but sometimes you want just 1 at a time and know thats what kind of data it is.
By design, zingLeafBitstrings dont have type other than if they're a hash and hashAlg.
This class is more of an example of a Lang than how I expect it to be used.
*/
public class DoubleLang implements Lang<Double>{

	public Double o(Zing z){
		//return z.d(z.isBig()?128:32);
		return z.d(32); //If its a single double, !isBig()
	}

	public Zing z(Double o){
		throw new Todo();
	}
	
	

}
