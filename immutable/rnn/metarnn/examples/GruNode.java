/** Ben F Rayfield offers this software opensource MIT license */
package immutable.rnn.metarnn.examples;
import immutable.rnn.metarnn.NodeType;

public class GruNode implements NodeType{

	/** (inData inGate forgetGate)->out */
	public int ins(){
		return 3;
	}

	/** (inData inGate forgetGate)->out */
	public int outs(){
		return 1;
	}

	public int mems(){
		return 0;
	}
	
	public boolean isDeterministic(){
		return true;
	}

	public void forward(float[] ins, float[] getOuts){
		throw new Error("TODO");
	}

	public void backprop(float[] getInDerivs, float[] ins, float[] outDerivs) {
		throw new Error("TODO");
	}

	@Override
	public void derivAt(float[][] getDerivs, float[] ins) {
		throw new Error("TODO");
	}

}
