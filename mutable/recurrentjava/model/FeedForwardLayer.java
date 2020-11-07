package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.autodiff.OpenclGraph;


public class FeedForwardLayer implements Model {

	private static final long serialVersionUID = 1L;
	//benrayfield made these public
	public Matrix W;
	public Matrix b;
	Unaflop f;
	
	public FeedForwardLayer(int inputDimension, int outputDimension, Unaflop f, float initParamsStdDev, Random rng) {
		W = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		b = new Matrix(outputDimension);
		this.f = f;
	}
	
	//benrayfield added this constructor
	public FeedForwardLayer(Matrix W, Matrix b, Unaflop f){
		this.W = W;
		this.b = b;
		this.f = f;
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) {
		//Matrix sum = g.add(g.mul(W, input), b);
		
		//FIXME call OpenclGraph.setIsTemp on the sum Matrix but not the out matrix
		
		int parallelSize = input.cols;
		Matrix sum = g.add_rowsCols_to_rowsColsWithColmult(g.mul(W, input), b, parallelSize);
		Matrix out = g.nonlin(f, sum);
		return out;
	}

	@Override
	public void resetState() {

	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		result.add(b);
		return result;
	}
}
