package immutable.rnn.metarnn.examples;

import immutable.rnn.metarnn.NodeType;
import immutable.util.MathUtil;

public class SigmoidNode implements NodeType{

	public int ins(){
		return 1;
	}

	public int outs(){
		return 1;
	}

	public int mems(){
		return 0;
	}
	
	public boolean isDeterministic() {
		return true;
	}

	public void forward(float[] ins, float[] getOuts){
		getOuts[0] = (float)MathUtil.sigmoid(ins[0]);
	}

	public void backprop(float[] getInDerivs, float[] ins, float[] outDerivs){
		
		//double out = MathUtil.sigmoid(ins[0]);
		//MathUtil.derivativeOfSigmoid(out);
	}

	@Override
	public void derivAt(float[][] getDerivs, float[] ins) {
		// TODO Auto-generated method stub
		
	}

}
