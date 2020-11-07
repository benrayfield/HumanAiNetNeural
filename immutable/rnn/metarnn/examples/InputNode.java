/** Ben F Rayfield offers this software opensource MIT license */
package immutable.rnn.metarnn.examples;

import immutable.rnn.metarnn.NodeType;

public class InputNode implements NodeType{

	public int ins(){
		return 0;
	}

	public int outs(){
		return 1;
	}
	
	public int mems(){
		return 0;
	}

	/** InputNode should be the only nondeterministic node type */
	public boolean isDeterministic(){
		return false;
	}

	public void forward(float[] ins, float[] getOuts){
		throw new UnsupportedOperationException("InputNode is nondeterministic");
	}

	public void backprop(float[] getInDerivs, float[] ins, float[] outDerivs){
		throw new UnsupportedOperationException("InputNode is nondeterministic");
	}

	@Override
	public void derivAt(float[][] getDerivs, float[] ins) {
		// TODO Auto-generated method stub
		
	}
	
	

}
