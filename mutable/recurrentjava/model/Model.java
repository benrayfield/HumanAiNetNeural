package mutable.recurrentjava.model;
import java.io.Serializable;
import java.util.List;

import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;


public interface Model extends Serializable {
	Matrix forward(Matrix input, Graph g);
	void resetState();
	List<Matrix> getParameters();
}
