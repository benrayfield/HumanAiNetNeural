package immutable.rnn.opencl;

import immutable.util.IntRange;
import immutable.util.Text;
import mutable.compilers.opencl.OpenclUtil;

/** Use ForestOp to get this working but laggy, then optimize ForestOp to use multiple ClMem and multiple opencl kernel
before returning to java. */
public class OpenclGru extends Gru{
	
	/*TODO optimize OpenclGru using immutable.compilers.opencl.ForestOp<T>,
	after getting CpuGru to learn the 5 mouse recordings in parallel
	like done by RecurrentJava (modified to learn in parallel),
	then do the same with OpenclGru, then tune OpenclGru
	to learn at least 100 of them in parallel.
	Use ForestOp only through OpenclUtil.callOpenclForest.
	*/
	
	//TODO use ForestOp.java, and optimize ForestOp to do multiple
	//opencl ndrange kernels at once in multiple clqueues with
	//dependnet order between them, each in their own CLMem
	//in the simplest case but reusing CLMem in some cases
	//when its known that CLMem wont be written to again
	//in the same forest of ForestOp.

	public OpenclGru(float[][][] weightHidden, float[][][] weightIn, float[][] biasHidden, IntRange outs){
		super(weightHidden, weightIn, biasHidden, outs);
	}
	
	public static String n = Text.n;
	
	//public static String openclNdrangeKernelCode_A =
		/*TODO
		"kernel void "+OpenclUtil.newKernelName()+"(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){"+n+
		"	int bd = get_global_id(0);"+n+
		"	bdOut[bd] = sum;"+n+
		"}";
		/*"kernel void "+OpenclUtil.newKernelName()+"(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){"+n+
		"	int bd = get_global_id(0);"+n+
		"	const int b = bd/dSize;"+n+ //TODO optimize allow get_global_id(more dims)?//
		"	const int d = bd%dSize;"+n+ //TODO optimize allow get_global_id(more dims)?
		"	float sum = 0;"+n+
		"	for(int c=0; c<cSize; c++){"+n+
		"		sum += bc[b*cSize+c]*cd[c*dSize+d];"+n+ //TODO optimize allow get_global_id(more dims)?
		"	}"+n+n+
		"	bdOut[bd] = sum;"+n+
		"}";*/

	public float[][] forward(float[][] nodeStates, float[][] ins, boolean returnAllNodes){
		if(!returnAllNodes) throw new Error("TODO");
		throw new Error("TODO use ForestOp.");
	}
	
	public float[][][] forwardN(
			float[][] firstNodeStates, float[][][] ins, boolean returnAllNodes){
		throw new Error("TODO");
	}

	public Gru learn(float[][] startNodeStates, LearnStep[] steps){
		throw new Error("TODO use ForestOp.");
	}

}
