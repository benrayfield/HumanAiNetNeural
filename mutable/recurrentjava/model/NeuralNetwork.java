package mutable.recurrentjava.model;

import java.util.ArrayList;
import java.util.List;

import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;

public class NeuralNetwork implements Model {

	private static final long serialVersionUID = 1L;
	
	//benrayfield made this public
	public List<Model> layers = new ArrayList<>();
	
	public NeuralNetwork(List<Model> layers) {
		this.layers = layers;
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) {
		Matrix prev = input;
		for (Model layer : layers) {
			prev = layer.forward(prev, g);
		}
		return prev;
	}

	@Override
	public void resetState() {
		for (Model layer : layers) {
			layer.resetState();
		}
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		for (Model layer : layers) {
			result.addAll(layer.getParameters());
		}
		return result;
	}
}
