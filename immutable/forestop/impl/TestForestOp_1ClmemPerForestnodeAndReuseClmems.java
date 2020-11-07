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

public class TestForestOp_1ClmemPerForestnodeAndReuseClmems{
	
	public static void main(String[] args){
		
		
		
		/*TODO useTheMatmulOptimizationIn(TestOpenclLocalMem.java)WhichDoubledSpeedToMultiply57PairsOf1024x1024FloatMatrixPerSecond
		
		trySmallCubesOfMatmulInLocalMemoryToExtremelyReduceGlobalMemoryReads[
			Try copying 2 small squares of the 2 matrixs into local memory (such as 16kB. fill it
			with 4kB*2 of data from each of 2 matrixs). Think of matrix multiply as a cube
			with 3 2d matrixs on 3 adjacent sides sharing a corner. The 2 horizontal sides are the
			input matrix, and the bottom face is the output matrix. Compute a small cube inside the
			big cube in each localMemory. Then sum the vertically aligning small cubes. This should
			make it alot faster cuz without it theres cubed number of reads from global memory, and
			its said in nvidia opencl optimization doc that local memory is 100 times lower lag than
			global memory. Copying 32x32 floats from each of 2 arrays into local memory, which sums
			to 8kB, should do about 32 times less reads from global memory in total so should be 32
			times faster. I'm getting 50gflops, so it should get 1600gflops, if this theory works
			out. TRY IT ASAP, after learning how to use localMemory.
			
			How much local memory?
			https://www.nvidia.com/content/GTC/documents/1068_GTC09.pdf says some cards have 16kb "shared memory"
			and some have 48k. Is that opencl local memory?
			
			Or is it already doing that from the longer loop doing a dotProduct per thread?
					
					https://www.reddit.com/r/gpgpu/comments/h94m5y/in_opencl_for_reducing_reads_of_global_memory_how/
						
						
			So far I've been doing 1 row in a CLMem and 1 column in another CLMem per 1 gpu thread, the slow way.
			I havent coded the 32x32 thing yet. There would be no write conflicts. Each address is written only
			once. How do I copy from global memory to local memory just once, and each of up to 1024 gpu threads
			(times number of 32x32x32 volumes, so if its 2 1024x1024 matrixs, that would
					be 1024*1024*1024/32=33554432 gpu threads) would read 64 of the 2048 floats then write 1
			float to another global CLMem? I need a different number of gpu threads to copy from global to local,
					than to read from that local, cuz for example each row or column in local is read by up to 32
					gpu threads. That could be too much data to write globally, being 32 times bigger than the
					output array, so maybe the temp 32x32 output could be stored in each local memory while
						another kernel call copies the next lower 32x32 squares (so repeat this 32 times)
						to local memory?
								
			maybe all I need to do is get_local_id in 2d and get_global_id in 2d and define 32x32 blocks as
			workgroup and let it cache that into local memory if it does that automatically?
		]*/
		
		/*TODO testcases for how to call this,
		since I dont think I've finished the interface,
		and testcases will help me figure out whats missing.
		Use this to build a simple RNN that runs n states in parallel
		each 20 neural cycles, back and forth between 2 CLMem for
		the neural states like doubleBuffering, with another CLMem
		for the constant weights, and display 1 of the neural states
		over time in a ColorFlowFromCenterColumn.
		*/
		
		//wrap in JFrame ColorFlowFromCenterColumn ui = TODO;
		JComponent ui = null;
		/** do this many neural cycles (each at least 1 opencl ndrange kernel)
		before opencl returns to java, for lower lag.
		*/
		//int cycles = 30;
		//int cycles = 1000;
		int cycles = 64; //FIXME why is nodeStatesBuf.get(5)=0.0 (should be nonzero) when cycles>4? Am I not making it wait for one kernel to finish before starting the next?
		//int cycles = 100;
		//int cycles = 1;
		//int parallelSize = 500;
		//int parallelSize = 1000;
		//int parallelSize = 64*512;
		int parallelSize = 64*512;
		//int parallelSize = 100;
		//int nodes = 50;
		//int nodes = 500;
		int nodes = 64*8;
		//int nodes = 100;
		float newWeightAve = 0, newWeightDev = .2f;
		
		float[] weights = new float[nodes*nodes];
		for(int i=0; i<weights.length; i++){
			//weights[i] = newWeightAve+newWeightDev*(float)Rand.strongRand.nextGaussian();
			weights[i] = newWeightAve+newWeightDev*(float)Rand.weakRand.nextGaussian();
		}
		float[] nodeStates = new float[parallelSize*nodes];
		for(int i=0; i<nodeStates.length; i++){
			//nodeStates[i] = (float)Rand.strongRand.nextFloat(); //0 to 1
			nodeStates[i] = (float)Rand.weakRand.nextFloat(); //0 to 1
		}
		
		
		while(true){ //repeat blocks of neuralCycles, displaying after each.
			//nodeStates = nodeStatesAfterCycles_lowLevelWithoutUsingForesting(
			//	weights, nodeStates, cycles, parallelSize, nodes);
			nodeStates = nodeStatesAfterCycles_lowLevelWithoutUsingForesting(
				weights, nodeStates, cycles, parallelSize, nodes);
			//TODO
			//copy to array in ui (use Mem to do that?);
			//ui.repaint();
			lg("nodeStates[0]="+nodeStates[0]+" TODO display each as pixel brightness in ColorFlowFromCenterColumn");
		}
	}
	
	public static final int MAX_CYCLES_AT_ONCE = 10000;
	
	static final String n = "\n";
	
	/*
	public static float[] nodeStatesAfterCycles(float[] weights, float[] firstNodeStates, int cycles, int parallelSize, int nodes){
		if(firstNodeStates.length != parallelSize*nodes) throw new Error(
			"firstNodeStates.length="+firstNodeStates.length+" parallelSize="+parallelSize+" nodes="+nodes);
		if(weights.length != nodes*nodes) throw new Error(
			"weights.length="+weights.length+" nodes="+nodes);
		if(MAX_CYCLES_AT_ONCE < cycles) throw new Error(
			"MAX_CYCLES_AT_ONCE="+MAX_CYCLES_AT_ONCE+" cycles="+cycles);
		Foresting f = new OpenclForesting();
		//f.close(); //can be done multiple times
		//like doubleBuffering. Alternating neuralCycles use 1 or the other.
		//Each of these is same size as firstNodeStates.
		Mem nodeStatesA = f.newMem(MemType.gpuMem, firstNodeStates);
		Mem nodeStatesB = null;
		Read nodeStates = writeForNodeStatesA.read();
		Read otherNodeStates = TODO;//writeForNodeStatesA.read();
		
		
		Mem weightsMem = f.newMem(MemType.gpuMem,weights);
		Read weightsRead = TODO;
		//TODO copy input
		Read parallelSizeReadInt = f.newRead(parallelSize);
		Read nodesReadInt = f.newRead(nodes);
		for(int i=0; i<cycles; i++){
			//call rnnCode on [nodeStatesA or nodeStatesB],
			//reading one and writing the other, and read weights.
			Write w = f.newWrite(nodeStates, rnnCode,
				weightsRead, otherNodeStates, parallelSizeReadInt, nodesReadInt);
			nodeStates = w.read();
			
			Read temp = nodeStates;
			nodeStates = otherNodeStates;
			otherNodeStates = temp;
		}
		//TODO copy to output
		f.run();
		
	}*/

	/** This will contain code similar to whats in OpenclUtil
	to implement nodeStatesAfterCycles as an experiment
	before Foresting, Read, Write, ForestOp, etc are working.
	This
	*/
	public static float[] nodeStatesAfterCycles_lowLevelWithoutUsingForesting(float[] weights, float[] firstNodeStates, int cycles, int parallelSize, int nodes){
		if(firstNodeStates.length != parallelSize*nodes) throw new Error(
			"firstNodeStates.length="+firstNodeStates.length+" parallelSize="+parallelSize+" nodes="+nodes);
		if(weights.length != nodes*nodes) throw new Error(
			"weights.length="+weights.length+" nodes="+nodes);
		if(MAX_CYCLES_AT_ONCE < cycles) throw new Error(
			"MAX_CYCLES_AT_ONCE="+MAX_CYCLES_AT_ONCE+" cycles="+cycles);
		
		int flopsPerLoopBody = 2;
		//int flopsPerLoopBody = 4;
		double flopsPerCycle = parallelSize*nodes*flopsPerLoopBody;
		double flops = 0;
		
		Lwjgl l = Lwjgl.instance();
		
		/** https://community.khronos.org/t/calling-the-same-kernel-object-multiple-times/1340/3
		"clSetKernelArg does change the argument values of the kernel. clEnqueueNDRangeKernel will
		enqueue a kernel execution with the corresponding argument values associated with the
		kernel when clEnqueueNDRangeKernel is called.
		A call to clSetKernelArg does not impact the kernel argument values used for prior
		clEnqueueNDRangeKernel calls, only future clEnqueueNDRangeKernel calls."
		*/
		
		lg("rnnCode="+rnnCode_modifiedForTestDontUseForRnn);
		CompiledKernel k = l.compiledOrFromCache(rnnCode_modifiedForTestDontUseForRnn);
		
		FloatBuffer nodeStatesBuf = BufferUtils.createFloatBuffer(firstNodeStates.length);
		lg("START: firstNodeStates[5]="+firstNodeStates[5]);
		lg("START: weights[17]="+weights[17]);
		
		boolean allowCopyhostptr = false; //FIXME if false this uses 0s instead of weights and first node states, as a test
		lg("allowCopyhostptr="+allowCopyhostptr);
		
		CLMem clWeights = allowCopyhostptr ? l.copy(weights) : l.newClmemReadableAndWritable(weights.length*4);
		//CLMem clNodeStatesA = l.copy(firstNodeStates);
		//CLMem clNodeStatesB = l.copy(firstNodeStates); //TODO optimize. only copy this once.
		/*FloatBuffer fbNodeStatesA = BufferUtils.createFloatBuffer(firstNodeStates.length);
		FloatBuffer fbNodeStatesB = BufferUtils.createFloatBuffer(firstNodeStates.length);
		CLMem clNodeStatesA = l.clmemWrapsJavaMem(fbNodeStatesA);
		CLMem clNodeStatesB = l.clmemWrapsJavaMem(fbNodeStatesB);
		*/
		
		//FloatBuffers fbs[] = new FloatBuffer[0];
		//CLMem clmems = new CLMem[cycles];
		long totalCycles = 0;
		
		//FIXME free globalWorkSize?
		//int nd = 1;
		int nd = 2;
		PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(nd);
		globalWorkSize.put(0, parallelSize);
		//int locXAndY = 32;
		//globalWorkSize.put(0, parallelSize/locX);
		//globalWorkSize.put(1, TODO);
		//PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(nd);
		//localWorkSize.put(0, locXAndY);
		//localWorkSize.put(1, locXAndY);
		CLCommandQueue q = l.queue();
		double timeStart = 0;
		boolean even = true;
		CLMem[] clmems = new CLMem[cycles+1];
		boolean eachCycleGetsItsOwnClmem = true;
		//boolean eachCycleGetsItsOwnClmem = false;
		for(int i=0; i<clmems.length; i++){
			if(allowCopyhostptr && i == 0){
				clmems[i] = l.copy(firstNodeStates); //FIXME this doesnt remember across the while loop, but its just a speed test
			}else{
				if(i<80 || eachCycleGetsItsOwnClmem){
				//if(i<2 || eachCycleGetsItsOwnClmem){
					clmems[i] = l.newClmemReadableAndWritable(firstNodeStates.length*4);
				}else{
					//clmems[i] = clmems[i-2];
					clmems[i] = clmems[i-80];
				}
			}
		}
		while(true){
			//CL10.clEnqueueBarrier(q);
			CLMem clNodeStatesIn = null;
			CLMem clNodeStatesOut = null;
			for(int i=0; i<cycles; i++){
				//CL10.clEnqueueBarrier(q);
				even = !even; //not same as cycles being even since that repeats in while loop.
				//clNodeStatesIn = even ? clNodeStatesA : clNodeStatesB;
				//clNodeStatesOut = even ? clNodeStatesB : clNodeStatesA;
				clNodeStatesIn = clmems[i];
				clNodeStatesOut = clmems[i+1];
				k.kernel.setArg(0, clNodeStatesOut);
				k.kernel.setArg(1, clWeights);
				k.kernel.setArg(2, clNodeStatesIn);
				k.kernel.setArg(3, parallelSize);
				k.kernel.setArg(4, nodes);
				CL10.clEnqueueNDRangeKernel(q, k.kernel, nd, null, globalWorkSize, null, null, null);
				//CL10.clEnqueueNDRangeKernel(q, k.kernel, nd, null, globalWorkSize, localWorkSize, null, null);
				//CL10.clEnqueueBarrier(q);
			}
			if(clNodeStatesOut != null){
				//CL10.clEnqueueBarrier(q);
				nodeStatesBuf.rewind();
				CL10.clEnqueueReadBuffer(q, clNodeStatesOut, CL10.CL_TRUE, 0, nodeStatesBuf, null, null);
			}
			//CL10.clEnqueueBarrier(q);
			
			//FIXME? why is clEnqueueReadBuffer needed with CL_MEM_USE_HOST_PTR?
			//FloatBuffer lastOutBuf = clNodeStatesOut==clNodeStatesA ? fbNodeStatesA : fbNodeStatesB;
			//CL10.clEnqueueReadBuffer(q, clNodeStatesOut, CL10.CL_TRUE, 0, lastOutBuf, null, null);
			
			CL10.clFinish(q);
			totalCycles += cycles;
			double now = Time.now();
			if(timeStart == 0){
				timeStart = now;
			}else{
				flops += cycles*flopsPerCycle;
			}
			double duration = now-timeStart;
			//lg("totalCycles="+totalCycles+" gflop/sec="+((flops*1e-9)/(now-timeStart)));
			//String nsb = ""+lastOutBuf.get(5);
			String nsb = ""+nodeStatesBuf.get(5);
			lg("END: totalCycles="+totalCycles+" cyc/sec="+(totalCycles/duration)+" gflop/sec="+((flops*1e-9)/duration)+" nodeStatesBuf.get(5)="+nsb);
			
			//TODO free CLMems (including globalWorkSize?)
			
			//TODO try CL_MEM_USE_HOST_PTR instead of CL_MEM_COPY_HOST_PTR,
			//and use CLMem and FloatBuffer that refer to the same memory in some cases?
			
			
		}
	}
	
	public static final String rnnCode_modifiedForTestDontUseForRnn =
		//"openclNdrangeKernel:"+n+
		"//out[parallelSize*nodes] nodeStates[parallelSize*nodes] weights[nodes*nodes]"+n+
		"kernel void "+OpenclUtil.newKernelName()+"(global float* out, global const float* weights, global const float* nodeStates, const int parallelSize, const int nodes){"+n+
		//"kernel void "+OpenclUtil.newKernelName()+"(local float* out, local const float* weights, local const float* nodeStates, const int parallelSize, const int nodes){"+n+
		"	const int pn = get_global_id(0);"+n+
		//todo get_group_id
		"	const int parallelIndex = pn/nodes;"+n+ //TODO optimize allow get_global_id(more dims)?
		"	const int nodeTo = pn%nodes;"+n+ //TODO optimize allow get_global_id(more dims)?
		"	float sum = 0;"+n+ //TODO bias
		"	int offsetA = nodeTo*nodes;"+n+
		//todo get_local_id(0) and get_local_id(1) and 
		"	int offsetB = parallelIndex*nodes;"+n+
		"	for(int nodeFrom=0; nodeFrom<nodes; nodeFrom++){"+n+
		"		sum += weights[offsetA+nodeFrom]*nodeStates[offsetB+nodeFrom];"+n+
		"	}"+n+
		"	float chance = 1/(1+exp(-sum));"+n+
		"	float one = 1;"+n+
		"	float randFraction = fmod(fabs(sum)*49999,one);"+n+
		"	float c = ceil(chance-randFraction);"+n+
		"	float z = 0;"+n+
		//strangely, on new computer (nvidia card instead of amd apu), fmax has to be 2 float vars
		//or 2 constants but cant be a constant and a float var.
		"	float weightedCoinFlip = fmax(z,c);"+n+
		//"	float weightedCoinFlip = fmax(0,ceil(chance-randFraction));"+n+
		//"	out[pn] = 2.000001+weightedCoinFlip+nodeStates[pn];"+n+ //FIXME remove the 2.000001 and nodeStates thats just there for testing
		"	out[pn] = nodeStates[pn]+1.001+weightedCoinFlip;"+n+ //FIXME remove this line, use the weightedCoinFlip line
		//"	out[pn] = weightedCoinFlip;"+n+
		//"	out[pn] = 1/(1+exp(-sum));"+n+
		//"	out[pn] = nodeStates[pn]+1;"+n+
		"}";
	
	/** TODO use OpenclProgs.matmulCode1dAs2dThenSigmoid instead? *
	public static final String rnnCode_modifiedForTestDontUseForRnn =
		//"openclNdrangeKernel:"+n+
		"//out[parallelSize*nodes] nodeStates[parallelSize*nodes] weights[nodes*nodes]"+n+
		"kernel void "+OpenclUtil.newKernelName()+"(global float* out, global const float* weights, global const float* nodeStates, const int parallelSize, const int nodes){"+n+
		//"kernel void "+OpenclUtil.newKernelName()+"(local float* out, local const float* weights, local const float* nodeStates, const int parallelSize, const int nodes){"+n+
		"	const int pn = get_global_id(0);"+n+
		todo get_group_id
		"	const int parallelIndex = pn/nodes;"+n+ //TODO optimize allow get_global_id(more dims)?
		"	const int nodeTo = pn%nodes;"+n+ //TODO optimize allow get_global_id(more dims)?
		"	float sum = 0;"+n+ //TODO bias
		"	int offsetA = nodeTo*nodes;"+n+
		todo get_local_id(0) and get_local_id(1) and 
		"	int offsetB = parallelIndex*nodes;"+n+
		"	for(int nodeFrom=0; nodeFrom<nodes; nodeFrom++){"+n+
		"		sum += weights[offsetA+nodeFrom]*nodeStates[offsetB+nodeFrom];"+n+
		"	}"+n+
		"	float chance = 1/(1+exp(-sum));"+n+
		"	float randFraction = fmod(fabs(sum)*49999,1);"+n+
		"	float c = ceil(chance-randFraction);"+n+
		"	float z = 0;"+n+
		//strangely, on new computer (nvidia card instead of amd apu), fmax has to be 2 float vars
		//or 2 constants but cant be a constant and a float var.
		"	float weightedCoinFlip = fmax(z,c);"+n+
		//"	float weightedCoinFlip = fmax(0,ceil(chance-randFraction));"+n+
		//"	out[pn] = 2.000001+weightedCoinFlip+nodeStates[pn];"+n+ //FIXME remove the 2.000001 and nodeStates thats just there for testing
		"	out[pn] = nodeStates[pn]+1.001+weightedCoinFlip;"+n+ //FIXME remove this line, use the weightedCoinFlip line
		//"	out[pn] = weightedCoinFlip;"+n+
		//"	out[pn] = 1/(1+exp(-sum));"+n+
		//"	out[pn] = nodeStates[pn]+1;"+n+
		"}";
	*/

}

