package mutable.recurrentjava.loss;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.recurrentjava.matrix.Matrix;

public class LossSumOfSquares implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) {
		FSyMem actualOutputDw = actualOutput.mem("dw");
		FSyMem actualOutputW = actualOutput.mem("w");
		FSyMem targetOutputW = targetOutput.mem("w");
		for (int i = 0; i < targetOutput.size; i++) {
			float errDelta = actualOutputW.get(i) - targetOutputW.get(i);
			actualOutputDw.putPlus(i, errDelta);
		}
	}
	
	@Override
	public float measure(Matrix actualOutput, Matrix targetOutput) {
		float sum = 0;
		FSyMem actualOutputW = actualOutput.mem("w");
		FSyMem targetOutputW = targetOutput.mem("w");
		for (int i = 0; i < targetOutput.size; i++) {
			float errDelta = actualOutputW.get(i) - targetOutputW.get(i);
			sum += 0.5f * errDelta * errDelta;
		}
		return sum;
	}
}
