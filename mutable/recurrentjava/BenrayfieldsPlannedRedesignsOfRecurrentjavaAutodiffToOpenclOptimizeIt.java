package mutable.recurrentjava;

public class BenrayfieldsPlannedRedesignsOfRecurrentjavaAutodiffToOpenclOptimizeIt {
	
	/*
	DONE: ~2019-11 upgraded recurrentjava GruLayer to learn
	multiple dataseqs in parallel,
	and its working for 5 parallel of mouseY recordings.
	
	DONE: convert recurrentjava gru from doubles to floats and verify it still works.
	
	DONE: test lag and speed of 1 opencl kernel both reading and writing
	the same CLMem. In TestForestOp I temporarily
	used a different rnnCode that adds a constant to the nodeStates CLMem
	and the speed and lag were the same.
	This only wrote to same index in doublebuffer, in both buffers at once.
	
	UPDATE 2020-11-7: planning to use lazycl in OpenclGraph.java (sibling of CpuGraph.java)
		instead of dealing with opencl directly in OpenclGraph.
	
	Despite it not using opencl's full speed
	cuz of extra IO between every small math op (such as elmul),
	I'm going to start by opencl optimizing the autodiff,
	so in theory the existing interface of recurrentjava,
	which I have slightly modified to make it work in parallel,
	will be opencl optimized.
	I will do this by replacing the double[]s in Matrix
	with FloatBuffer and CLMem (each array will have both),
	and the stepCache array is only used in Matrixs
	that are part of the getParameters() of a Model,
	not the middle calculations, so dont allocate those.
	DoubleBuffer is also supported
	in some hardware, but not older hardware,
	and floats are much faster, so I'm going with floats.
	...
	This will require changes in Matrix, Graph, and Trainer.
	...
	I'll make a subclass of Graph called OpenclGraph,
	and leave the existing Graph doing it by cpu.
	OpenclGraph will delay the calculations
	until they're all ready, then do it all in 1 call of opencl,
	except Trainer.updateModelParams maybe should be a separate call.
	...
	Since allocation of CLMem is slow,
	I'll pool them similar to the WeakHashMap<ForestOp,Mem>
	where Mem contains CLMem,
	but will have to do it differently than keying by ForestOp
	cuz many Matrix are created just for 1 use
	so cant optimize by remembing which CLMem they used.
	Instead will have to pool by allocating and freeing
	CLMems by their size, and by freeing I mean
	the CLMem still exists but at the java level
	its marked as available for another Matrix to use.
	...
	Theres alot of += in recurrentjava,
	and I'm allocating CLMems as readwrite,
	but I've not yet ever written a kernel that
	both reads and writes the same memory.
	If that can be done, and if it doesnt slow things too much,
	then just use += directly in the kernels,
	else doubleBuffer in 2 CLMems.
	...
	Each pair<FloatBuffer,CLMem>, if allocated with
	USE_HOST_POINTER (spelling?), still needs
	CLQueue to copy from CLMem to that FloatBuffer
	for java to read it in the FloatBuffer.
	...
	Some parts maybe I'll keep in CLMem until observe in java if ever,
	like neural weights.
	...
	Add a func to Matrix: makeReadable()
	that queues a copy from CLMem to FloatBuffer,
	of the multiple CLMems and multiple FloatBuffers in that Matrix.
	Matrix will no longer contain double[],
	so maybe I should create a new class and use it instead of matrix?
	Or maybe I should convert Matrix to completely floatbuffer
	and only in a subclass also have CLMem?
	Java will call makeReadable() on the Matrixs that java
	wants to read after opencl does all the calculations at once,
	then java will read it from the FloatBuffers in that matrix.
	...
	Test if recurrentjava still learns with floats
	(probably it will since I see it jumping around and still learning)
	since it originally had doubles. I know it works
	when I replaced just matmul with floats.
	If doubles are needed but hardware doesnt support it,
	they can be emulated by multiple ints or longs or maybe by
	3 floats if careful about the low bits
	but probably ints are the best way to emulate it.
	
	The fastest way would be to represent neuralnode
	(which may be GruNode or LstmNode etc)
	as 2 opencl kernels, one forward and one backprop.
	The forward one computes the outputs, given the inputs
	to that node.
	The backprop one computes the derivatives
	at the inputs and derivatives at the outputs,
	given the inputs which it uses to compute the outputs
	then backprops from them to the inputs.
	This way is not autodiff.
	BUT this way is harder to build, so I'm starting
	with optimizing autodiff.
	There is value in autodiff since experiments
	can be created and redesigned much faster.
	After experiments are working well,
	can further optimize by doing it this way.
	*/
	

}


