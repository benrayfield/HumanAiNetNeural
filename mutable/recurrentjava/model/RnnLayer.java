package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;


public class RnnLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	
	Matrix W, b;
	
	Matrix context;
	
	Unaflop f;
	
	public RnnLayer(int inputDimension, int outputDimension, Unaflop hiddenUnit, float initParamsStdDev, Random rng) {
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.f = hiddenUnit;
		W = Matrix.rand(outputDimension, inputDimension+outputDimension, initParamsStdDev, rng);
		b = new Matrix(outputDimension);
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g){
		
		Matrix concat = g.concatVectors(input, context);
		
		Matrix sum = g.mul(W, concat);
		sum = g.add(sum, b);
		Matrix output = g.nonlin(f, sum);
		
		//rollover activations for next iteration
		context = output;
		
		return output;
	}

	@Override
	public void resetState() {
		context = new Matrix(outputDimension);
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		result.add(b);
		return result;
	}

}
