package mutable.recurrentjava;
import java.io.File;
import java.util.Random;

import immutable.recurrentjava.flop.unary.LinearUnit;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.recurrentjava.util.NeuralNetworkHelper;
import mutable.util.Files;
import mutable.recurrentjava.datasets.TextGeneration;
import mutable.recurrentjava.datastructs.DataSet;

public class ExamplePaulGraham2 {
	public static void main(String[] args) throws Exception {
		
		/*
		 * Character-by-character sentence prediction and generation, closely following the example here:
		 * http://cs.stanford.edu/people/karpathy/recurrentjs/
		*/
		
		

		//start benrayfield changed
		//String textSource = "PaulGraham";
		String textSource = "PaulGrahamSmall";
		String path = new File(Files.dirWhereThisProgramStarted, "data/recurrentjava/"+textSource+".txt").getAbsolutePath();
		//DataSet data = new TextGeneration("datasets/text/"+textSource+".txt");
		DataSet data = new TextGeneration(path);
		//String savePath = "saved_models/"+textSource+".ser";
		String savePath = new File(Files.dirWhereThisProgramStarted, "data/recurrentjava/ser/"+textSource+".ser").getAbsolutePath();
		Files.delete(new File(savePath));
		//end benrayfield changed
		
		boolean initFromSaved = true; //set this to false to start with a fresh model
		boolean overwriteSaved = true;
		
		TextGeneration.reportSequenceLength = 100;
		TextGeneration.singleWordAutocorrect = false; //set this to true to constrain generated sentences to contain only words observed in the training data.

		//int bottleneckSize = 10; //one-hot input is squeezed through this
		int hiddenDimension = 200;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;
		
		Random rng = new Random();
		/*
		//benrayfield commented this replaced with calling gru
		Model neuralnet = NeuralNetworkHelper.makeLstmWithInputBottleneck( 
				data.inputDimension, bottleneckSize, 
				hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		*/
		int parallelSize = 1;
		Model neuralnet = NeuralNetworkHelper.makeLstm(
			parallelSize,
			data.inputDimension,
			hiddenDimension, hiddenLayers, 
			data.outputDimension, data.getModelOutputUnitToUse(), 
			initParamsStdDev, rng);
		
		/*Model neuralnet = NeuralNetworkHelper.makeGru(
			data.inputDimension, 
			hiddenDimension, hiddenLayers, 
			data.outputDimension, data.getModelOutputUnitToUse(), 
			initParamsStdDev, rng);
		*/
		
		/*Model neuralnet = NeuralNetworkHelper.makeRnn(
				data.inputDimension, 
				hiddenDimension, hiddenLayers, 
				data.outputDimension, new LinearUnit(), data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		*/
		
		
		int reportEveryNthEpoch = 10;
		
		//int trainingEpochs = 1000; //benrayfield changed
		int trainingEpochs = 100; //benrayfield changed
		
		
		Trainer.train(trainingEpochs, learningRate, neuralnet, data, reportEveryNthEpoch, initFromSaved, overwriteSaved, savePath, rng);
		
		System.out.println("done.");
	}
}
