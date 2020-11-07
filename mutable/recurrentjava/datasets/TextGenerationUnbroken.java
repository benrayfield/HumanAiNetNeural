package mutable.recurrentjava.datasets;
import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.recurrentjava.flop.unary.LinearUnit;
import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.util.Util;
import mutable.recurrentjava.loss.LossSoftmax;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;


public class TextGenerationUnbroken extends DataSet {

	private static final long serialVersionUID = 1L;
	public static int reportSequenceLength = 100;
	public static boolean reportPerplexity = true;
	private static Map<String, Integer> charToIndex = new HashMap<>();
	private static Map<Integer, String> indexToChar = new HashMap<>();
	private static int dimension;
	
	public static String generateText(Model model, int steps, boolean argmax, float temperature, Random rng) throws Exception {
		Matrix start = new Matrix(dimension);
		model.resetState();
		//Graph g = new Graph(false);
		Graph g = new CpuGraph(false);
		Matrix input = (Matrix)start.clone();
		FSyMem inputW = input.mem("w");
		String result = "";
		for (int s = 0; s < steps; s++) {
			Matrix logprobs = model.forward(input, g);
			Matrix probs = LossSoftmax.getSoftmaxProbs(logprobs, temperature);
			FSyMem probsW = probs.mem("w");
			
			int indxChosen = -1;
			if (argmax) {
				float high = Float.NEGATIVE_INFINITY;
				for (int i = 0; i < probs.size; i++) {
					if (probsW.get(i) > high) {
						high = probsW.get(i);
						indxChosen = i;
					}
				}
			}
			else {
				indxChosen = Util.pickIndexFromRandomVector(probs, rng);
			}
			String ch = indexToChar.get(indxChosen);
			result += ch;
			for (int i = 0; i < input.size; i++) {
				inputW.put(i, 0);
			}
			inputW.put(indxChosen, 1f);
		}
		result = result.replace("\n", "\"\n\t\"");
		return result;
	}
	
	public TextGenerationUnbroken(String path, int totalSequences, int sequenceMinLength, int sequenceMaxLength, Random rng) throws Exception {
		
		System.out.println("Text generation task");
		System.out.println("loading " + path + "...");
		
		File file = new File(path);
		List<String> lines_ = Files.readAllLines(file.toPath(), Charset.defaultCharset());
		
		String text = "";
		for (String line : lines_) {
			text += line + "\n";
		}
		
		Set<String> chars = new HashSet<>();
		int id = 0;
		
		System.out.println("Characters:");
		
		System.out.print("\t");
		
		for (int i = 0; i < text.length(); i++) {
			String ch = text.charAt(i) + "";
			if (chars.contains(ch) == false) {
				if (ch.equals("\n")) {
					System.out.print("\\n");
				}
				else {
					System.out.print(ch);
				}
				chars.add(ch);
				charToIndex.put(ch, id);
				indexToChar.put(id, ch);
				id++;
			}
		}
		System.out.println("");
		
		dimension = chars.size();
		
		List<DataSequence> sequences = new ArrayList<>();
		
		for (int s = 0; s < totalSequences; s++) {
			List<float[]> vecs = new ArrayList<>();
			int len = rng.nextInt(sequenceMaxLength - sequenceMinLength + 1) + sequenceMinLength;
			int start = rng.nextInt(text.length() - len);
			for (int i = 0; i < len; i++) {
				String ch = text.charAt(i+start) + "";
				int index = charToIndex.get(ch);
				float[] vec = new float[dimension];
				vec[index] = 1f;
				vecs.add(vec);
			}
			DataSequence sequence = new DataSequence();
			for (int i = 0; i < vecs.size() - 1; i++) {
				sequence.steps.add(new DataStep(vecs.get(i), vecs.get(i+1)));
			}
			sequences.add(sequence);
		}

		System.out.println("Total unique chars = " + chars.size());
		
		training = sequences;
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		inputDimension = sequences.get(0).steps.get(0).input.size;
		int loc = 0;
		while (sequences.get(0).steps.get(loc).targetOutput == null) {
			loc++;
		}
		outputDimension = sequences.get(0).steps.get(loc).targetOutput.size;
	}

	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		System.out.println("========================================");
		System.out.println("REPORT:");
		if (reportPerplexity) {
			System.out.println("\ncalculating perplexity over entire data set...");
			double perplexity = LossSoftmax.calculateMedianPerplexity(model, training);
			System.out.println("\nMedian Perplexity = " + String.format("%.4f", perplexity));
		}
		float[] temperatures = {1f, 0.75f, 0.5f, 0.25f, 0.1f};
		for (float temperature : temperatures) {
			System.out.println("\nTemperature "+temperature+" prediction:");
			String guess = TextGenerationUnbroken.generateText(model, reportSequenceLength, false, temperature, rng);
			System.out.println("\t\"..." + guess + "...\"");
		}
		System.out.println("\nArgmax prediction:");
		String guess = TextGenerationUnbroken.generateText(model, reportSequenceLength, true, 1f, rng);
		System.out.println("\t\"..." + guess + "...\"");
		System.out.println("========================================");
	}

	@Override
	public Unaflop getModelOutputUnitToUse() {
		return new LinearUnit();
	}
}
