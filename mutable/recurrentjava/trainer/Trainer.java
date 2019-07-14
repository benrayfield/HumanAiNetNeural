package mutable.recurrentjava.trainer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

import mutable.recurrentjava.util.FileIO;
import mutable.recurrentjava.RjOptions;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;

public class Trainer {
	
	public static double decayRate = 0.999;
	public static double smoothEpsilon = 1e-8;
	public static double gradientClipValue = 5;
	public static double regularization = 0.000001; // L2 regularization strength
	
	public static double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, Random rng) throws Exception {
		return train(trainingEpochs, learningRate, model, data, reportEveryNthEpoch, false, false, null, rng);
	}
	
	/** benrayfield added this */
	public static final Consumer<Model> defaultStateResetter = (Model m)->m.resetState();
	
	public static final Consumer<Matrix> ignoreMatrix = (Matrix m)->{};
	
	public static double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, boolean initFromSaved, boolean overwriteSaved, String savePath, Random rng) throws Exception {
		System.out.println("--------------------------------------------------------------");
		if (initFromSaved) {
			System.out.println("initializing model from saved state...");
			try {
				model = (Model)FileIO.deserialize(savePath);
				data.DisplayReport(model, rng);
			}
			catch (Exception e) {
				System.out.println("Oops. Unable to load from a saved state.");
				System.out.println("WARNING: " + e.getMessage());
				System.out.println("Continuing from freshly initialized model instead.");
			}
		}
		double result = 1.0;
		for (int epoch = 0; epoch < trainingEpochs; epoch++) {
			
			String show = "epoch["+(epoch+1)+"/"+trainingEpochs+"]";
			
			double reportedLossTrain = pass(ignoreMatrix, defaultStateResetter, learningRate, model, data.training, true, data.lossTraining, data.lossReporting);
			result = reportedLossTrain;
			if (Double.isNaN(reportedLossTrain) || Double.isInfinite(reportedLossTrain)) {
				throw new Exception("WARNING: invalid value for training loss. Try lowering learning rate.");
			}
			double reportedLossValidation = 0;
			double reportedLossTesting = 0;
			if (data.validation != null) {
				reportedLossValidation = pass(ignoreMatrix, defaultStateResetter, learningRate, model, data.validation, false, data.lossTraining, data.lossReporting);
				result = reportedLossValidation;
			}
			if (data.testing != null) {
				reportedLossTesting = pass(ignoreMatrix, defaultStateResetter, learningRate, model, data.testing, false, data.lossTraining, data.lossReporting);
				result = reportedLossTesting;
			}
			show += "\ttrain loss = "+String.format("%.5f", reportedLossTrain);
			if (data.validation != null) {
				show += "\tvalid loss = "+String.format("%.5f", reportedLossValidation);
			}
			if (data.testing != null) {
				show += "\ttest loss  = "+String.format("%.5f", reportedLossTesting);
			}
			System.out.println(show);
			
			if (epoch % reportEveryNthEpoch == reportEveryNthEpoch - 1) {
				data.DisplayReport(model, rng);
			}
			
			if (overwriteSaved) {
				FileIO.serialize(savePath, model);
			}
			
			if (reportedLossTrain == 0 && reportedLossValidation == 0) {
				System.out.println("--------------------------------------------------------------");
				System.out.println("\nDONE.");
				break;
			}
		}
		return result;
	}

	/** benrayfield added the Consumer params */
	public static double pass(Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			double learningRate, Model model, List<DataSequence> sequences, boolean applyTraining,
			Loss lossTraining, Loss lossReporting) throws Exception{
		
		double numerLoss = 0;
		double denomLoss = 0;
		
		for (DataSequence seq : sequences) {
			//benrayfield added param stateResetter so can start at random state to reduce overfitting model.resetState();
			stateResetter.accept(model);
			Graph g = new Graph(applyTraining);
			for (DataStep step : seq.steps) {
				Matrix output = model.forward(step.input, g);
				outputListener.accept(output); //benrayfield added this to avoid recomputing it in UnidimView
				if (step.targetOutput != null) {
					double loss = lossReporting.measure(output, step.targetOutput);
					//benrayfield: System.out.println("pass loss="+loss);
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						return loss;
					}
					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(output, step.targetOutput);
					}
				}
			}
			List<DataSequence> thisSequence = new ArrayList<>();
			thisSequence.add(seq);
			if (applyTraining) {
				g.backward(); //backprop dw values
				if(!RjOptions.testDelayedUpdateOfWeights){
					updateModelParams(model, learningRate); //update params
				}else{
					System.out.println("WARNING: testDelayedUpdateOfWeights skips call of updateModelParams, make sure to do at end of batch.");
				}
			}	
		}
		return numerLoss/denomLoss;
	}
	
	public static void updateModelParams(Model model, double stepSize) throws Exception {
		for (Matrix m : model.getParameters()) {
			for (int i = 0; i < m.w.length; i++) {
				
				// rmsprop adaptive learning rate
				double mdwi = m.dw[i];
				m.stepCache[i] = m.stepCache[i] * decayRate + (1 - decayRate) * mdwi * mdwi;
				
				// gradient clip
				if (mdwi > gradientClipValue) {
					mdwi = gradientClipValue;
				}
				if (mdwi < -gradientClipValue) {
					mdwi = -gradientClipValue;
				}
				
				// update (and regularize)
				m.w[i] += - stepSize * mdwi / Math.sqrt(m.stepCache[i] + smoothEpsilon) - regularization * m.w[i];
				m.dw[i] = 0;
				
				/*benrayfield
				FIXME how can I testDelayedUpdateOfWeights when weightChange is decaying m.w?
				Also delay the processing of m.dw?
				I only want to delay until the end of a batch.
				In the parallel code, the weight arrays are shared and the node states etc
				are parallelSize times bigger.
				Since weight arrays are shared, how much is that affecting backprop?
				Maybe I can just call updateModelParams at end of batch.
				*/
				
			}
		}
	}
}
