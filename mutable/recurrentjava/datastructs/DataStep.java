package mutable.recurrentjava.datastructs;
import java.io.Serializable;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.recurrentjava.matrix.Matrix;


public class DataStep implements Serializable {

	private static final long serialVersionUID = 1L;
	public Matrix input = null;
	public Matrix targetOutput = null;
	
	public DataStep() {
		
	}
	
	public DataStep(float[] input, float[] targetOutput) {
		this.input = new Matrix(input);
		if (targetOutput != null) {
			this.targetOutput = new Matrix(targetOutput);
		}
	}
	
	@Override
	public String toString() {
		String result = "";
		int end = input.buf("w").capacity();
		FSyMem inputW = input.mem("w");
		for (int i = 0; i < end; i++) {
			result += String.format("%.5f", inputW.get(i)) + "\t";
		}
		result += "\t->\t";
		if (targetOutput != null) {
			FSyMem targetOutputW = targetOutput.mem("w");
			for (int i = 0; i < targetOutputW.mem().capacity(); i++) {
				result += String.format("%.5f", targetOutputW.get(i)) + "\t";
			}
		}
		else {
			result += "___\t";
		}
		return result;
	}
}
