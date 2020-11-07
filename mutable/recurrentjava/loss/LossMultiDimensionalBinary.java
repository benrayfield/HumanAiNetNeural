package mutable.recurrentjava.loss;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.recurrentjava.matrix.Matrix;

public class LossMultiDimensionalBinary implements Loss {

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
		
		FSyMem actualOutputW = actualOutput.mem("w");
		FSyMem targetOutputW = targetOutput.mem("w");
		
		for (int i = 0; i < targetOutput.size; i++) {
			if (targetOutputW.get(i) >= 0.5 && actualOutputW.get(i) < 0.5) {
				return 1;
			}
			if (targetOutputW.get(i) < 0.5 && actualOutputW.get(i) >= 0.5) {
				return 1;
			}
		}
		return 0;
	}

}
