package mutable.recurrentjava.loss;
import java.util.ArrayList;
import java.util.List;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.util.Util;


public class LossSoftmax implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix logProbs, Matrix targetOutput) {
		int targetIndex = getTargetIndex(targetOutput);
		Matrix probs = getSoftmaxProbs(logProbs, 1f);
		FSyMem logProbsDw = logProbs.mem("dw");
		FSyMem probsW = probs.mem("w");
		for (int i = 0; i < probsW.size; i++){
			logProbsDw.put(i, probsW.get(i));
		}
		logProbsDw.putPlus(targetIndex, -1f);
	}

	@Override
	public float measure(Matrix logprobs, Matrix targetOutput){
		int targetIndex = getTargetIndex(targetOutput);
		Matrix probs = getSoftmaxProbs(logprobs, 1f);
		FSyMem probsW = probs.mem("w");
		float cost = (float) -Math.log(probsW.get(targetIndex));
		return cost;
	}

	public static float calculateMedianPerplexity(Model model, List<DataSequence> sequences){
		float temperature = 1f;
		List<Float> ppls = new ArrayList<>();
		for (DataSequence seq : sequences) {
			float n = 0;
			float neglog2ppl = 0;
			
			//Graph g = new Graph(false);
			Graph g = new CpuGraph(false); //FIXME? if using OpenclGraph other places, dont create a new one, reuse same DependnetBuilder.
			model.resetState();
			for (DataStep step : seq.steps) {
				Matrix logprobs = model.forward(step.input, g);
				Matrix probs = getSoftmaxProbs(logprobs, temperature);
				FSyMem probsW = probs.mem("w");
				int targetIndex = getTargetIndex(step.targetOutput);
				float probOfCorrect = probsW.get(targetIndex);
				float log2prob = (float)(Math.log(probOfCorrect)/Math.log(2)); //change-of-base
				neglog2ppl += -log2prob;
				n += 1;
			}
			
			n -= 1; //don't count first symbol of sentence
			float ppl = (float)Math.pow(2, (neglog2ppl/(n-1)));
			ppls.add(ppl);
		}
		return Util.median(ppls);
	}
	
	public static Matrix getSoftmaxProbs(Matrix logprobs, float temperature){	
		Matrix probs = new Matrix(logprobs.size);
		FSyMem logprobsW = logprobs.mem("w");
		if (temperature != 1.0) {
			for (int i = 0; i < logprobs.size; i++) {
				logprobsW.putDivide(i, temperature);
			}
		}
		float maxval = Float.NEGATIVE_INFINITY;
		FSyMem probsW = probs.mem("w");
		for (int i = 0; i < logprobs.size; i++) {
			if (logprobsW.get(i) > maxval) {
				maxval = logprobsW.get(i);
			}
		}
		float sum = 0;
		for (int i = 0; i < logprobs.size; i++) {
			probsW.put(i, (float)Math.exp(logprobsW.get(i) - maxval)); //all inputs to exp() are non-positive
			sum += probsW.get(i);
		}
		for (int i = 0; i < probsW.size; i++) {
			probsW.putDivide(i, sum);
		}
		return probs;
	}

	private static int getTargetIndex(Matrix targetOutput){
		FSyMem targetOutputW = targetOutput.mem("w");
		for (int i = 0; i < targetOutput.size; i++) {
			if (targetOutputW.get(i) == 1f) {
				return i;
			}
		}
		throw new Error("no target index selected");
	}
}
