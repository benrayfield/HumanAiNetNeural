package mutable.recurrentjava.loss;

import java.io.Serializable;

import mutable.recurrentjava.matrix.Matrix;

public interface Loss extends Serializable {
	void backward(Matrix actualOutput, Matrix targetOutput) throws Exception;
	double measure(Matrix actualOutput, Matrix targetOutput) throws Exception;
}
