package mutable.recurrentjava.loss;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.recurrentjava.matrix.Matrix;

public class LossArgMax implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) {
		throw new UnsupportedOperationException();
		
	}

	@Override
	public float measure(Matrix actualOutput, Matrix targetOutput) {
		if (actualOutput.size != targetOutput.size) {
			throw new Error("mismatch");
		}
		//double maxActual = Double.NEGATIVE_INFINITY;
		//double maxTarget = Double.NEGATIVE_INFINITY;
		float maxActual = Float.NEGATIVE_INFINITY;
		float maxTarget = Float.NEGATIVE_INFINITY;
		int indxMaxActual = -1;
		int indxMaxTarget = -1;
		FSyMem actualOutputW = actualOutput.mem("w");
		FSyMem targetOutputW = targetOutput.mem("w");
		for (int i = 0; i < actualOutputW.size; i++) {
			if (actualOutputW.get(i) > maxActual){
				maxActual = actualOutputW.get(i);
				indxMaxActual = i;
			}
			if (targetOutputW.get(i) > maxTarget) {
				maxTarget = targetOutputW.get(i);
				indxMaxTarget = i;
			}
		}
		if (indxMaxActual == indxMaxTarget) {
			return 0f;
		}
		else {
			return 1f;
		}
	}
	
}
