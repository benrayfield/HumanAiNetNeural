package mutable.recurrentjava.trainer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.rnn.RnnParams;
import mutable.recurrentjava.util.FileIO;
import mutable.compilers.opencl.BiMem;
import mutable.compilers.opencl.FMem;
import mutable.recurrentjava.RjOptions;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;

public class Trainer {
	
	/*public static double decayRate = 0.999;
	public static double smoothEpsilon = 1e-8;
	public static double gradientClipValue = 5;
	public static double regularization = 0.000001; // L2 regularization strength
	*/
	
	public static float train(RnnParams params, int trainingEpochs, Model model, DataSet data, int reportEveryNthEpoch, Random rng) throws Exception {
		return train(params, trainingEpochs, model, data, reportEveryNthEpoch, false, false, null, rng);
	}
	
	/** benrayfield added this */
	public static final Consumer<Model> defaultStateResetter = (Model m)->m.resetState();
	
	public static final Consumer<Matrix> ignoreMatrix = (Matrix m)->{};
	
	public static float train(RnnParams params, int trainingEpochs, Model model, DataSet data, int reportEveryNthEpoch, boolean initFromSaved, boolean overwriteSaved, String savePath, Random rng) throws Exception {
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
		float result = 1f;
		for (int epoch = 0; epoch < trainingEpochs; epoch++) {
			
			String show = "epoch["+(epoch+1)+"/"+trainingEpochs+"]";
			
			float reportedLossTrain = pass(params, ignoreMatrix, defaultStateResetter, model, data.training, true, data.lossTraining, data.lossReporting);
			result = reportedLossTrain;
			if (Float.isNaN(reportedLossTrain) || Float.isInfinite(reportedLossTrain)) {
				throw new Exception("WARNING: invalid value for training loss. Try lowering learning rate.");
			}
			float reportedLossValidation = 0;
			float reportedLossTesting = 0;
			if (data.validation != null) {
				reportedLossValidation = pass(params, ignoreMatrix, defaultStateResetter, model, data.validation, false, data.lossTraining, data.lossReporting);
				result = reportedLossValidation;
			}
			if (data.testing != null) {
				reportedLossTesting = pass(params, ignoreMatrix, defaultStateResetter, model, data.testing, false, data.lossTraining, data.lossReporting);
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
	public static float pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			Model model, List<DataSequence> sequences, boolean applyTraining,
			Loss lossTraining, Loss lossReporting){
		
		//FIXME this should do all DataSequence in parallel
		boolean parallel = true;
		
		if(parallel){
			
			float numerLoss = 0;
			float denomLoss = 0;
			
			int countSteps = sequences.get(0).steps.size();
			
			//benrayfield added param stateResetter so can start at random state to reduce overfitting model.resetState();
			stateResetter.accept(model);
			Graph g = new CpuGraph(applyTraining);
			int inputSizePerStep = sequences.get(0).steps.get(0).input.cols; //FIXME row vs col?
			int outputSizePerStep = sequences.get(0).steps.get(0).targetOutput.cols; //FIXME row vs col?
			
			//for (DataSequence seq : sequences) {
			for (int stepNum=0; stepNum<countSteps; stepNum++){
				//for (DataStep step : seq.steps) {
				
					
					//these are steps all at the same time.
					//The earlier and later steps are looped around this (TODO)
					Matrix inputOfAllSteps = new Matrix(inputSizePerStep, sequences.size()); //FIXME rows vs cols backward?
					Matrix correctOutputOfAllSteps = new Matrix(outputSizePerStep, sequences.size()); //FIXME rows vs cols backward?
					for(int seqNum=0; seqNum<sequences.size(); seqNum++){
						DataStep step = sequences.get(seqNum).steps.get(stepNum);
						//FIXME since I'm using only 1 input and 1 output, might have got rows vs cols backward
						//System.arraycopy(
						FMem.arraycopy(
							step.input.buf("w"), 0, //copy from (all)
							inputOfAllSteps.buf("w"), seqNum*inputSizePerStep, //copy to (range)
							inputSizePerStep);
						//System.arraycopy(
						FMem.arraycopy(
							step.targetOutput.buf("w"), 0, //copy from (all)
							correctOutputOfAllSteps.buf("w"), seqNum*outputSizePerStep, //copy to (range)
							outputSizePerStep);
						
					}
					Matrix output = model.forward(inputOfAllSteps, g);
					//if(output.rows != correctOutputOfAllSteps.rows || output.cols != correctOutputOfAllSteps.cols)
					//	throw new Error("output and correctOutputOfAllSteps are diff sizes");
					//Matrix output = model.forward(step.input, g);
					outputListener.accept(output); //benrayfield added this to avoid recomputing it in UnidimView
					
					float loss = lossReporting.measure(output, correctOutputOfAllSteps);
					//benrayfield: System.out.println("pass loss="+loss);
					if(Float.isNaN(loss) || Float.isInfinite(loss)) {
						throw new Error("loss is not finite: "+loss);
						//return loss;
					}
					numerLoss += loss;
					denomLoss++;			
					if(applyTraining) {
						lossTraining.backward(output, correctOutputOfAllSteps);
					}
					
					/*for(int seqNum=0; seqNum<sequences.size(); seqNum++){
					
						//FIXME should these things happen once per dataseq or once per batch?
						//For now let it do one per step, and just see if it does anything
						//on screen since Ive changed enough code its hard to keep track.
						
						DataStep step = sequences.get(seqNum).steps.get(stepNum);
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
					}*/
				//}
				//List<DataSequence> thisSequence = new ArrayList<>();
				//thisSequence.add(seq);
			}
			
			((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
			
			if (applyTraining) {
				//g.doTasksInCpu(); //backprop dw values
				//if(!RjOptions.testDelayedUpdateOfWeights){
					updateModelParams(params, model);
				//}else{
				//	System.out.println("WARNING: testDelayedUpdateOfWeights skips call of updateModelParams, make sure to do at end of batch.");
				//}
			}
			
			((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
			
			return numerLoss/denomLoss;
			
		}else{
		
			float numerLoss = 0;
			float denomLoss = 0;
			
			for (DataSequence seq : sequences) {
				//benrayfield added param stateResetter so can start at random state to reduce overfitting model.resetState();
				stateResetter.accept(model);
				Graph g = new CpuGraph(applyTraining);
				for (DataStep step : seq.steps) {
					Matrix output = model.forward(step.input, g);
					outputListener.accept(output); //benrayfield added this to avoid recomputing it in UnidimView
					if (step.targetOutput != null) {
						float loss = lossReporting.measure(output, step.targetOutput);
						//benrayfield: System.out.println("pass loss="+loss);
						if (Float.isNaN(loss) || Float.isInfinite(loss)) {
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
				
				((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
				
				if (applyTraining) {
					//g.doTasksInCpu(); //backprop dw values
					//if(!RjOptions.testDelayedUpdateOfWeights){
						updateModelParams(params, model);
					//}else{
					//	System.out.println("WARNING: testDelayedUpdateOfWeights skips call of updateModelParams, make sure to do at end of batch.");
					//}
				}
				
				((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
			}
			return numerLoss/denomLoss;
		}
	}
	
	public static void updateModelParams(RnnParams p, Model model){
		for (Matrix m : model.getParameters()) {
			FSyMem mW = m.mem("w");
			FSyMem mDw = m.mem("dw");
			FSyMem mStepCache = m.mem("stepCache");
			for (int i = 0; i < m.size; i++) {
				
				// rmsprop adaptive learning rate
				float mdwi = mDw.get(i);
				//m.stepCache[i] = m.stepCache[i] * p.rjTrainerDecayRate + (1 - p.rjTrainerDecayRate) * mdwi * mdwi;
				mStepCache.put(i, mStepCache.get(i) * p.rjTrainerDecayRate + (1 - p.rjTrainerDecayRate) * mdwi * mdwi);
				
				// gradient clip
				if (mdwi > p.rjTrainerGradientClipValue) {
					mdwi = p.rjTrainerGradientClipValue;
				}
				if (mdwi < -p.rjTrainerGradientClipValue) {
					mdwi = -p.rjTrainerGradientClipValue;
				}
				
				// update (and regularize)
				//m.w[i] += - p.learnRate * mdwi / Math.sqrt(m.stepCache[i] + p.rjTrainerSmoothEpsilon) - p.rjTrainerRegularization * m.w[i];
				float mwi = mW.get(i);
				mW.put(i, (float)(mwi - p.learnRate * mdwi / Math.sqrt(mStepCache.get(i) + p.rjTrainerSmoothEpsilon) - p.rjTrainerRegularization * mwi));
				mDw.put(i,0);
				
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
