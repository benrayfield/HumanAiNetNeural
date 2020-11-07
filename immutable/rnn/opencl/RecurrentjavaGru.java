package immutable.rnn.opencl;

import immutable.util.IntRange;

public class RecurrentjavaGru extends Gru{
	
	public RecurrentjavaGru(float[][][] weightHidden, float[][][] weightIn, float[][] biasHidden, IntRange outs){
		super(weightHidden, weightIn, biasHidden, outs);
	}

	public float[][] forward(float[][] nodeStates, float[][] ins, boolean returnAllNodes){
		if(!returnAllNodes) throw new Error("TODO");
		throw new Error("TODO");
	}
	
	/** this will avoid the inefficiency of copying between recurrentjava and immutable form
	every step.
	*/
	public float[][][] forwardN(
			float[][] firstNodeStates, float[][][] ins, boolean returnAllNodes){
		throw new Error("TODO");
	}

	public Gru learn(float[][] startNodeStates, LearnStep[] steps) {
		// TODO Auto-generated method stub
		return null;
	}

}
