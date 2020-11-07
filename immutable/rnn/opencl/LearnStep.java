package immutable.rnn.opencl;

/** v vecs of input and v vecs of output.
Normally a small fraction of nodes are outNodes, such as 2 for mouseY and mouseX chaostime ahead of inputs,
and these start at hiddenNodeIndex Gru.outIndex or Lstm.outIndex.
*/
public class LearnStep{
	
	public final float[][] in, correctOut;
	
	public final float learnRate;
	
	public LearnStep(float[][] in, float[][] correctOut, float learnRate){
		this.in = in;
		this.correctOut = correctOut;
		this.learnRate = learnRate;
	}
	
	public LearnStep learnRate(float f){
		if(f == learnRate) return this;
		return new LearnStep(in, correctOut, f);
	}

}
