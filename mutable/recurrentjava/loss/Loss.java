package mutable.recurrentjava.loss;

import java.io.Serializable;

import mutable.recurrentjava.matrix.Matrix;

public interface Loss extends Serializable {
	void backward(Matrix actualOutput, Matrix targetOutput);
	float measure(Matrix actualOutput, Matrix targetOutput);
}
