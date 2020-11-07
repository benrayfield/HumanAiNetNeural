package mutable.recurrentjava;

public class RjOptions{
	private RjOptions(){}
	
	/** When benrayfield found recurrentjava (derived from recurrentjs) ~Y2019M3 it was not openCL optimized
	so is modifying Recurrentjava to call his OpenclUtil which calls LWJGL.
	If this var is true it does that, else does the original recurrentjava code.
	The opencl optimization only helps when theres multiple vecs (parallelIndex) learned/predicted at once,
	so LstmLayer.cellContext and .hiddenContext would have dims of nodesXparallelIndex
	(or is it parallelIndexXnodes?) compared to recurrentjava used parallel size 1 (Matrix.cols=1),
	or maybe RecurrentJava is already doing that and I didnt see it.
	That means a DataSequence is as many DataSequences as parallelIndex (0 to parallelSize-1)
	and all are the same number of time steps.
	I might have some of these dims backward but I'll figure it out and maybe forget to update this comment.
	<br><br>
	My usecase of the many datasequences of the same size
	will be randomly select ranges (partially overlapping in some cases)
	from a time window of maybe the last minute, to learn mouse movements of multiple people at once
	and look for and influence the creation of patterns between them as mmgMouseai game
	will use many such sparse LSTMs (and other kinds of AI) fit together (by pairs of inputs being
	b/(b+c) and b+c is similar to continuous neuralDropout) to navigate the sparse p2p network of MMG AI games. 
	I might have some of these dims backward but I'll figure it out and maybe forget to update this comment.
	<br><br>
	Also theres some good code in AMD's C++ opencl API, but this is java.
	<br><br>
	OpenclUtil doesnt modify any of the inputs or outputs, even if marked as mutable like "global double* bdOut".
	OpenclUtil.callOpencl takes an opencl kernel code string and caches its compiled form,
	and takes an nd-range int[1] (only 1d works so far, but wraps 2d array in 1d and back automatically)
	and Object[] params which may be array, int, etc,
	and returns an Object[] of the same size as Object[] params, reusing those not modified,
	and replacing those modified by opencl. It knows the difference by basic parsing of parts of the kernel string.
	I will upgrade it to do doubles instead of just floats, for use in recurrentjava,
	but Ive read that opencl is not reliable of support for doubles but is reliable of support for floats,
	and similar for its reliability of support for determinism (to the extent that it can be merkle hashed).
	*
	public static boolean opencl = false;
	//public static boolean opencl = true;
	*/
	
	/** an experiment to figure out whats wrong with the parallel code,
	by running the sequential code multiple times before updating the weights
	and see if it still learns well. If it doesnt learn well, maybe thats whats wrong
	with the parallel code, else the parallel code is broken some other way, as of 2019-5-10.
	*
	public static boolean testDelayedUpdateOfWeights = false;
	*/

}
