package immutable.rnn.opencl;

import java.util.Arrays;

import immutable.util.IntRange;
import immutable.util.MathUtil;

/** GRU neuralnet (TODO opencl optimized). TODO common interface between GRU and LSTM,
differing by that LSTM has 2 node state floats instead of GRU has 1.
This will be similar to the math in RecurrentJava,
but for opencl I couldnt reuse any of that code, just the standard math
(or at least thats how it appears 2019-9).
I'm still learning about gru and lstm so I'm going to opencl optimize GRU first,
then LSTM, then refactor to common code between them or maybe
I'll only use LSTM going forward (and later expand to backprop qlearning through it).
<br><br>
I might implement this 2 ways, 1 as recurrentjava subclass and 1 as opencl subclass.
<br><br>
Uses ForestOp for multiple opencl kernels at once to reduce lag,
but forestop code isnt well opencl optimized as of 2019-9 so alternates between
opencl and java for each opencl kernel. That will be optimized after get GRU working. 
*/
public abstract class Gru{
	
	public static final int GRU_MIX = 0;
	public static final int GRU_RESET = 1;
	public static final int GRU_NEW = 2;
	
	/*Do I want to generalize to layers such as recurrent lstm (stateful)
	then feedforwardQlearn (stateless except lstm's state)?
	Or do I want to keep it as just 1 recurrent layer (the same lstm or gru as that example's first layer)?
	A big difference between this software and recurrentjava is how the math will be clustered
	for opencl optimization, especially doing a whole inner node logic in 1 kernel
	intstead of recurrentjava does that as multiple matrix ops.
	I'll start with just recurrent gru and work up to more advanced things.
	
	weight
	bias (only for hidden?)
	
	external
	hidden
	
	inData
	inGate
	forgetGate
	
	TODO should inputs and outputs be separate from nodestates?
	
	
	TODO funcs to predict and learn, for batch of dataseq and for 1 step ahead
	from starting nodestates.
	*/
	

	
	
	
	
	
	
	
	
	
	
	
	
	//TODO choose [from][to] vs [to][from]
			
			
	/*	
	FIXME where do rmsprop etc weight norming floats go,
	like the stepCache etc in recurrentjava Trainer.java?
	Maybe those should go in separate class since Gru is not a learning
	process but is a snapshot of what has already been learned.
	*/
			
			
			
			
			
	
	/** [toNodeType][from][to]. From includes inputs. TODO swap the [from] and [to] dimIndexs? */
	public final float[][][] weightHidden;
	
	/** [toNodeType][from][to] */
	public final float[][][] weightIn;
	
	/** [toNodeType][to] */
	public final float[][] biasHidden;
	
	public final int nodes;
	
	/*TODO float[...] weightIn and weightHidden, considering that I might only want bias on hiddens but not inputs.
	TODO bias.
	*/
	
	/** Which hiddenNodeIndex is output index 0 and consecutive from there. Examples: 0, size-outs. */
	public final IntRange outs;
	
	
	public Gru(float[][][] weightHidden, float[][][] weightIn, float[][] biasHidden, IntRange outs){
		this.weightHidden = weightHidden;
		this.weightIn = weightIn;
		this.biasHidden = biasHidden;
		this.outs = outs;
		nodes = biasHidden[0].length;
		if(
			weightHidden[0].length != nodes
			|| weightHidden[0][1].length != nodes
			|| weightIn[0].length != nodes
			|| weightIn[1][0].length != nodes
			|| outs.start < 0 || nodes < outs.endExclusive
		) throw new Error("wrong size or range");
	}
	
	/** same as forward(...) but only the output nodes, starting at outOffset is return's 0 index (TODO which dimIndex). */
	//TODO adjust arrays num of dims: public abstract float[][] out(float[][][] nodeStates, float[][] in);
	
	/** predict 1 step forward for v vecs of nodestates.
	<br><br>
	firstNodeStates[vec][node]
 	in[vec][inNode]
 	returnFloat[vec][node]
 	If returnAllNodes then returns states of all nodes, else only the output nodes.
 	*/
	public abstract float[][] forward(float[][] firstNodeStates, float[][] in, boolean returnAllNodes);
	
	/** predict n steps forward for v vecs of nodestates. n=ins.length.
	<br><br>
	firstNodeStates[vec][node]
 	ins[step][vec][inNode]
 	returnFloat[step][vec][node]
 	If returnAllNodes then returns states of all nodes, else only the output nodes.
 	*/
	public abstract float[][][] forwardN(
		float[][] firstNodeStates, float[][][] ins, boolean returnAllNodes);
	
	/** TODO some func like this (diff params probably) after get the 1 cycle forward(...) working.
	returnFloat[][][] *
	*
	public abstract ? forward(float[][] startNodeStates, float[][] ins, int cycles);
	*/
	
	/**
	startNodeStates[vec][node]
	<br><br>
	ins[cycle][vec][inNode]
	<br><br>
	Normally a small fraction of nodes are outNodes, such as 2 for mouseY and mouseX chaostime ahead of inputs.
	//correctOuts[cycle][vec][outNode]
	<br><br>
	Normally flat or increasing, since startNodeStates should be at least partially random,
	and converge toward a pattern based on inputs.
	learnRatePerCycle[cycle]
	*/
	public abstract Gru learn(float[][] startNodeStates, LearnStep[] steps);

}


