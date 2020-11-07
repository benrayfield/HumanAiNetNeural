/** Ben F Rayfield offers this software opensource MIT license */
package immutable.rnn.metarnn.examples;
import immutable.rnn.metarnn.NodeType;

/** TODO the feeding back into itself might need a change to NodeType */
public class LstmNode implements NodeType{

	/** (inData inGate forgetGate outGate mem out)->(mem out).
	FIXME that might be too many as the left out might be derived from the others.
	*/
	public int ins(){
		return 6;
	}

	/** (inData inGate forgetGate outGate mem out)->(mem out)
	FIXME that might be too many as the left out might be derived from the others.
	*/
	public int outs() {
		return 2;
	}
	
	/** FIXME is this 1 or 2. */
	public int mems(){
		return 1;
	}
	
	public boolean isDeterministic(){
		return true;
	}

	public void forward(float[] ins, float[] getOuts){
		throw new Error("TODO");
	}

	public void backprop(float[] getInDerivs, float[] ins, float[] outDerivs){
		throw new Error("TODO");
	}

	@Override
	public void derivAt(float[][] getDerivs, float[] ins) {
		// TODO Auto-generated method stub
		
	}

}
