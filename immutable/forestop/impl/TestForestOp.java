package immutable.forestop.impl;

import static mutable.util.Lg.*;

import java.nio.FloatBuffer;

import javax.swing.JComponent;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.opencl.CLMem;

import immutable.forestop.Foresting;
import immutable.forestop.Mem;
import immutable.forestop.MemType;
import immutable.forestop.Read;
import immutable.forestop.Write;
import mutable.compilers.opencl.OpenclUtil;
import mutable.compilers.opencl.connectors.lwjgl.CompiledKernel;
import mutable.compilers.opencl.connectors.lwjgl.Lwjgl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;

public class TestForestOp {

	public static void main(String[] args) {

		/*
		 * TODO testcases for how to call this, since I dont think I've finished the
		 * interface, and testcases will help me figure out whats missing. Use this to
		 * build a simple RNN that runs n states in parallel each 20 neural cycles, back
		 * and forth between 2 CLMem for the neural states like doubleBuffering, with
		 * another CLMem for the constant weights, and display 1 of the neural states
		 * over time in a ColorFlowFromCenterColumn.
		 */

		// wrap in JFrame ColorFlowFromCenterColumn ui = TODO;
		JComponent ui = null;
		/**
		 * do this many neural cycles (each at least 1 opencl ndrange kernel) before
		 * opencl returns to java, for lower lag.
		 */
		// int cycles = 30;
		int cycles = 100;
		// int cycles = 1;
		int parallelSize = 500;
		// int nodes = 50;
		int nodes = 500;
		float newWeightAve = 0, newWeightDev = .2f;

		float[] weights = new float[nodes * nodes];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = newWeightAve + newWeightDev * (float) Rand.strongRand.nextGaussian();
		}
		float[] nodeStates = new float[parallelSize * nodes];
		for (int i = 0; i < nodeStates.length; i++) {
			nodeStates[i] = (float) Rand.strongRand.nextFloat(); // 0 to 1
		}

		while (true) { // repeat blocks of neuralCycles, displaying after each.
			// nodeStates = nodeStatesAfterCycles_lowLevelWithoutUsingForesting(
			// weights, nodeStates, cycles, parallelSize, nodes);
			nodeStates = nodeStatesAfterCycles_lowLevelWithoutUsingForesting(weights, nodeStates, cycles, parallelSize,
					nodes);
			// TODO
			// copy to array in ui (use Mem to do that?);
			// ui.repaint();
			lg("nodeStates[0]=" + nodeStates[0]
					+ " TODO display each as pixel brightness in ColorFlowFromCenterColumn");
		}
	}

	public static final int MAX_CYCLES_AT_ONCE = 10000;

	static final String n = "\n";

	/*
	 * public static float[] nodeStatesAfterCycles(float[] weights, float[]
	 * firstNodeStates, int cycles, int parallelSize, int nodes){
	 * if(firstNodeStates.length != parallelSize*nodes) throw new Error(
	 * "firstNodeStates.length="+firstNodeStates.length+" parallelSize="
	 * +parallelSize+" nodes="+nodes); if(weights.length != nodes*nodes) throw new
	 * Error( "weights.length="+weights.length+" nodes="+nodes);
	 * if(MAX_CYCLES_AT_ONCE < cycles) throw new Error(
	 * "MAX_CYCLES_AT_ONCE="+MAX_CYCLES_AT_ONCE+" cycles="+cycles); Foresting f =
	 * new OpenclForesting(); //f.close(); //can be done multiple times //like
	 * doubleBuffering. Alternating neuralCycles use 1 or the other. //Each of these
	 * is same size as firstNodeStates. Mem nodeStatesA = f.newMem(MemType.gpuMem,
	 * firstNodeStates); Mem nodeStatesB = null; Read nodeStates =
	 * writeForNodeStatesA.read(); Read otherNodeStates =
	 * TODO;//writeForNodeStatesA.read();
	 * 
	 * 
	 * Mem weightsMem = f.newMem(MemType.gpuMem,weights); Read weightsRead = TODO;
	 * //TODO copy input Read parallelSizeReadInt = f.newRead(parallelSize); Read
	 * nodesReadInt = f.newRead(nodes); for(int i=0; i<cycles; i++){ //call rnnCode
	 * on [nodeStatesA or nodeStatesB], //reading one and writing the other, and
	 * read weights. Write w = f.newWrite(nodeStates, rnnCode, weightsRead,
	 * otherNodeStates, parallelSizeReadInt, nodesReadInt); nodeStates = w.read();
	 * 
	 * Read temp = nodeStates; nodeStates = otherNodeStates; otherNodeStates = temp;
	 * } //TODO copy to output f.run();
	 * 
	 * }
	 */

	/**
	 * This will contain code similar to whats in OpenclUtil to implement
	 * nodeStatesAfterCycles as an experiment before Foresting, Read, Write,
	 * ForestOp, etc are working. This
	 */
	public static float[] nodeStatesAfterCycles_lowLevelWithoutUsingForesting(float[] weights, float[] firstNodeStates,
			int cycles, int parallelSize, int nodes) {
		if (firstNodeStates.length != parallelSize * nodes)
			throw new Error("firstNodeStates.length=" + firstNodeStates.length + " parallelSize=" + parallelSize
					+ " nodes=" + nodes);
		if (weights.length != nodes * nodes)
			throw new Error("weights.length=" + weights.length + " nodes=" + nodes);
		if (MAX_CYCLES_AT_ONCE < cycles)
			throw new Error("MAX_CYCLES_AT_ONCE=" + MAX_CYCLES_AT_ONCE + " cycles=" + cycles);

		int flopsPerLoopBody = 2;
		double flopsPerCycle = parallelSize * nodes * flopsPerLoopBody;
		double flops = 0;

		Lwjgl l = Lwjgl.instance();

		/**
		 * https://community.khronos.org/t/calling-the-same-kernel-object-multiple-times/1340/3
		 * "clSetKernelArg does change the argument values of the kernel.
		 * clEnqueueNDRangeKernel will enqueue a kernel execution with the corresponding
		 * argument values associated with the kernel when clEnqueueNDRangeKernel is
		 * called. A call to clSetKernelArg does not impact the kernel argument values
		 * used for prior clEnqueueNDRangeKernel calls, only future
		 * clEnqueueNDRangeKernel calls."
		 */

		// String rnnCode = rnnCodeAndAddSConstantIntoSameArrayWritten; //FIXME
		// commentout this

		lg("rnnCode=" + rnnCode);
		CompiledKernel k = l.compiledOrFromCache(rnnCode);

		FloatBuffer nodeStatesBuf = BufferUtils.createFloatBuffer(firstNodeStates.length);

		CLMem clWeights = l.copy(weights);
		// CLMem clNodeStatesA = l.copy(firstNodeStates);
		// CLMem clNodeStatesB = l.copy(firstNodeStates); //TODO optimize. only copy
		// this once.
		FloatBuffer fbNodeStatesA = BufferUtils.createFloatBuffer(firstNodeStates.length);
		FloatBuffer fbNodeStatesB = BufferUtils.createFloatBuffer(firstNodeStates.length);
		CLMem clNodeStatesA = l.clmemWrapsJavaMem(fbNodeStatesA);
		CLMem clNodeStatesB = l.clmemWrapsJavaMem(fbNodeStatesB);

		// FloatBuffers fbs[] = new FloatBuffer[0];
		// CLMem clmems = new CLMem[cycles];
		long totalCycles = 0;

		// FIXME free globalWorkSize?
		int nd = 1;
		PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(nd);
		globalWorkSize.put(0, parallelSize);
		CLCommandQueue q = l.queue();
		double timeStart = 0;
		boolean even = true;
		while (true) {
			CLMem clNodeStatesIn = null;
			CLMem clNodeStatesOut = null;
			/*
			 * CLMem[] clmems = new CLMem[cycles+1]; for(int i=0; i<clmems.length; i++){
			 * if(i == 0){ clmems[i] = l.copy(firstNodeStates); //FIXME this doesnt remember
			 * across the while loop, but its just a speed test }else{ clmems[i] =
			 * l.newClmemReadableAndWritable(firstNodeStates.length*4); } }
			 */
			for (int i = 0; i < cycles; i++) {

				even = !even; // not same as cycles being even since that repeats in while loop.
				clNodeStatesIn = even ? clNodeStatesA : clNodeStatesB;
				clNodeStatesOut = even ? clNodeStatesB : clNodeStatesA;
				// clNodeStatesIn = clmems[i];
				// clNodeStatesOut = clmems[i+1];
				k.kernel.setArg(0, clNodeStatesOut);
				k.kernel.setArg(1, clWeights);
				k.kernel.setArg(2, clNodeStatesIn);
				k.kernel.setArg(3, parallelSize);
				k.kernel.setArg(4, nodes);
				CL10.clEnqueueNDRangeKernel(q, k.kernel, nd, null, globalWorkSize, null, null, null);
			}
			if (clNodeStatesOut != null) {
				nodeStatesBuf.rewind();
				CL10.clEnqueueReadBuffer(q, clNodeStatesOut, CL10.CL_TRUE, 0, nodeStatesBuf, null, null);
			}

			// FIXME? why is clEnqueueReadBuffer needed with CL_MEM_USE_HOST_PTR?
			FloatBuffer lastOutBuf = clNodeStatesOut == clNodeStatesA ? fbNodeStatesA : fbNodeStatesB;
			CL10.clEnqueueReadBuffer(q, clNodeStatesOut, CL10.CL_TRUE, 0, lastOutBuf, null, null);

			CL10.clFinish(q);
			totalCycles += cycles;
			double now = Time.now();
			if (timeStart == 0) {
				timeStart = now;
			} else {
				flops += cycles * flopsPerCycle;
			}
			double duration = now - timeStart;
			// lg("totalCycles="+totalCycles+" gflop/sec="+((flops*1e-9)/(now-timeStart)));
			String nsb = "" + lastOutBuf.get(5);
			// String nsb = "TODO";
			lg("totalCycles=" + totalCycles + " cyc/sec=" + (totalCycles / duration) + " gflop/sec="
					+ ((flops * 1e-9) / duration) + " nodeStatesBuf.get(5)=" + nsb);

			// TODO free CLMems (including globalWorkSize?)

			// TODO try CL_MEM_USE_HOST_PTR instead of CL_MEM_COPY_HOST_PTR,
			// and use CLMem and FloatBuffer that refer to the same memory in some cases?
		}
	}

	/** TODO use OpenclProgs.matmulCode1dAs2dThenSigmoid instead? */
	public static final String rnnCode =
			// "openclNdrangeKernel:"+n+
			"//out[parallelSize*nodes] nodeStates[parallelSize*nodes] weights[nodes*nodes]" + n + "kernel void "
					+ OpenclUtil.newKernelName()
					+ "(global float* out, global const float* weights, global const float* nodeStates, const int parallelSize, const int nodes){"+ n +
					"	const int pn = get_global_id(0); //c" + n +
					"	const int parallelIndex = pn/nodes;" + n + // TODO
																													// optimize
																													// allow
																													// get_global_id(more
																													// dims)?
					"	const int nodeTo = pn%nodes;" + n + // TODO optimize allow get_global_id(more dims)?
					"	float sum = 0;" + n + // TODO bias
					"	int offsetA = nodeTo*nodes;" + n + "	int offsetB = parallelIndex*nodes;" + n +
					"	for(int nodeFrom=0; nodeFrom<nodes; nodeFrom++){" + n +
					"		sum += weights[offsetA+nodeFrom]*nodeStates[offsetB+nodeFrom];" + n +
					"	}" + n +
					"	float one = 1;" + n +
					"	float randFraction = fmod(fabs(sum)*49999,one);" + n +
					"	float chance = 1/(1+exp(-sum));" + n +
					"	float zero = 0;" + n +
					"	float weightedCoinFlip = fmax(zero,ceil(chance-randFraction));" + n +
					"	out[pn] = weightedCoinFlip;" + n +
					// " out[pn] = 1/(1+exp(-sum));"+n+
					// " out[pn] = nodeStates[pn]+1;"+n+
					"}";

	/** tests opencl reading and writing same memory in same kernel */
	public static final String rnnCodeAndAddSConstantIntoSameArrayWritten =
			// "openclNdrangeKernel:"+n+
			"//out[parallelSize*nodes] nodeStates[parallelSize*nodes] weights[nodes*nodes]" + n + "kernel void "
					+ OpenclUtil.newKernelName()
					+ "(global float* out, global const float* weights, global float* nodeStates, const int parallelSize, const int nodes){"
					+ n + "	const int pn = get_global_id(0); //b" + n + "	const int parallelIndex = pn/nodes;" + n + // TODO
																													// optimize
																													// allow
																													// get_global_id(more
																													// dims)?
					"	const int nodeTo = pn%nodes;" + n + // TODO optimize allow get_global_id(more dims)?
					"	float sum = 0;" + n + // TODO bias
					"	int offsetA = nodeTo*nodes;" + n + "	int offsetB = parallelIndex*nodes;" + n
					+ "	for(int nodeFrom=0; nodeFrom<nodes; nodeFrom++){" + n
					+ "		sum += weights[offsetA+nodeFrom]*nodeStates[offsetB+nodeFrom];" + n + "	}" + n
					+ "	float chance = 1/(1+exp(-sum));" + n + "	float randFraction = fmod(fabs(sum)*49999,1);" + n
					+ "	float weightedCoinFlip = fmax(0,ceil(chance-randFraction));" + n + "	nodeStates[pn] += 5;"
					+ n + // this is the change in rnnCodeAndAddSConstantIntoSameArrayWritten from rnnCode
					// " nodeStates[pn+nodeTo-parallelIndex] += 5;"+n+
					// " nodeStates[nodeTo*3+parallelIndex] += 6;"+n+
					// " nodeStates[pn] += .0001;"+n+ //tests opencl reading and writing same memory
					// in same kernel
					"	out[pn] = weightedCoinFlip;" + n +
					// " out[pn] = 1/(1+exp(-sum));"+n+
					// " out[pn] = nodeStates[pn]+1;"+n+
					"}";

}
