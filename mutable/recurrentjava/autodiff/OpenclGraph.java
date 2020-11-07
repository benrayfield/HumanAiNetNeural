package mutable.recurrentjava.autodiff;
import static immutable.util.ImmutableImportStatic.*;
import static mutable.util.MutableImportStatic.*;
import static mutable.compilers.opencl.TestOpencl.readLock;
import java.util.*;
import static mutable.compilers.opencl.TestOpencl.writeLock;
import static mutable.compilers.opencl.TestOpencl.readWriteLock;
import java.util.function.Consumer;
import java.util.function.Predicate;

import javax.sound.midi.MidiDevice.Info;

import immutable.acyclicflow.AcyclicFlowF;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.DependParam;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.Mem;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.ParallelSize;
import immutable.dependtask.DependOp;
import immutable.dependtask.DependnetBuilder;
import immutable.dependtask.LockState;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.rnn.RnnParams;
import immutable.util.ListUtil;
import mutable.compilers.opencl.FMem;
import mutable.compilers.opencl.OpenclUtil;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.FeedForwardLayer;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.model.NeuralNetwork;
import mutable.util.Dependencies;
import mutable.util.task.RunnableTask;
import mutable.util.task.Task;

/** A Graph implemented using OpenCL. A Graph is a mutable builder of numberCrunching ops with backprop (and in some cases also training) built in.
TODO: OpenCL optimized autodiff, including forward, backprop, and training,
such as learning in a recurrent GruLayer with 5 FeedForwardLayers above it.
<br><br>
See comment of OpenclGraph.doWork for how to use this as it has different requirements than CpuGraph.
Calculations are delayed and you can optimize them by setIsTemp.
<br><br>
Strictfp is just a semantic here since the calculations are actually done in opencl used with strict compile options
and should (in theory, todo verify in TestAutodiffCpuVsGpu*) compute the exact same bits.
<br><br>
As Runnable, it does the calculations, reading and writing in multiple Matrixs.
<br><br>
Cuz multiple opencl ndrange kernels are done at once before returning to CPU/java,
the contents of Matrixs (which each contain 3 lazyevaled FSyMems) are lazyEvaled,
but FIXME that lazyEval might not be lazy triggered and instead depend on some other code
calling OpenclUtil.callOpenclDependnet on some transform of OpenclGraph.dependnet()
which will be done in (TODO) OpenclGraph.eval() or something like that,
which will use the DependParams in the FSyMems (sy means DependParam as symbol)
matching that with the DependParams in the Set<DependOp> returned by dependnet(),
to know which FloatBuffers (in the FSymMems in the Matrixs) to copy between opencl and Matrix.
<br><br>
OLD...
<br><br>
TODO This delays calculation of Graph until its built
then does the whole thing in OpenclUtil.callOpenclForest(...).
<br><br>
This should run at low lag since on my computer opencl can do
1700 seqential ndrange kernels (one after the other) per second
when blocks of 30 kernels are done before returning to java,
or 100 kernels per second if returning to java after each kernel.
<br><br>
RecurrentJava's GruLayer and LstmLayer etc are wasteful
in creating array for every multiply and add, so
TODO write kernels that do the whole neuralnode in 1 kernel
(multiplied by how many time steps) and between that do matmul,
and write the backprop specific to each node type
based on the multiple inputs to each node (derive output value(s) of
that node from those inputs) and backpropping from the outputs 
<br><br>
Matrix is compatible with delayed eval (not as well organized as lazyeval),
or potentially lazyeval but leave that upgrade for later,
since it uses FSyMem objects which lazy create FloatBuffer
and their DependParam (SYmbol) is used in OpenclUtil.callOpenclDependnet
to refer to CLMem objects that most of them dont have FloatBuffer.
<br><br>
Every returned Matrix contains FSyMems whose FloatBuffer hasnt been filled yet.
Its filled all at once, among many Matrix, using the task lists in OpenclGraph.
*/
public strictfp class OpenclGraph implements Graph{
	
	/** immutable. FIXME make this null when op is added. */
	protected SortedSet<DependOp> cacheLastDependnetReturned;
	
	protected boolean applyBackprop;
	
	protected final List<DependOp> forwardpropOps = new ArrayList();
	
	protected final List<DependOp> backpropOpsInReverseOrder = new ArrayList();
	
	protected final List<DependOp> trainingpropOps = new ArrayList();
	
	/** After OpenclUtil.callOpenclDependnet, theres Mems to copy back into Matrixs.
	Each Matrix has 3 DependParams. For each of those, there can be a Consumer<Mem> which copies that back into that part of the Matrix.
	*
	protected final SortedMap<DependParam,Consumer<Mem>> writeMatrix = new TreeMap();
	*/
	protected final Set<Matrix> rememberAllMatrixs = new TreeSet();

	
	protected final Set<DependParam> isTemp = new HashSet();
	
	/** optional optimization (to not use, dont fill this) where the FloatBuffers in these Matrixs arent copied from opencl and are deleted
	by not telling OpenclUtil.callOpenclDependnet to output them, only to use them in internal calculations.
	Matrix contains 3 FSyMem (w dw and stepCache) which are each lazyEvaled. They start as just containing 3 DependParam (the sy(mbol)s).
	In the case of OpenclGraph (instead of CpuGraph), the FloatBuffers are never created for temp matrixs, instead creating CLMems for them,
	and for inputs and outputs they have both FloatBuffer and CLMem and its copied between.
	To prevent wastefully creating a duplicate buffer (CLMem and ALSO FloatBuffer) for temp Matrixs, add them to this Set<Matrix>.
	UPDATE: using DependParam instead of Matrix. 
	*/
	public void setIsTemp(DependParam dp, boolean isTemp){ this.isTemp.add(dp); }
	
	/** see setIsTemp */
	public boolean isTemp(DependParam dp){ return isTemp.contains(dp); }
	
	public OpenclGraph(boolean applyBackprop) {
		this.applyBackprop = applyBackprop;
	}
	
	public boolean applyBackprop(){ return applyBackprop; }
	
	public boolean isLazy(){ return true; }
	
	public boolean hasWork(){ return !forwardpropOps.isEmpty() || !backpropOpsInReverseOrder.isEmpty() || !trainingpropOps.isEmpty(); }
	
	/** Reads and writes the FloatBuffers in Matrixs except where OpenclGraph.setTemp(DependParam,true) of any of the 3 DependParams per Matrix.
	You use OpenclGraph to queue Matrix ops such as add and mul, which each return a lazyEval Matrix,
	and OpenclGraph.setIsTemp(DependParam,true) for temp calculations you dont want copied from opencl back to those Matrix parts,
	then OpenclGraph.doWork(), then get the results from the relevant Matrixs such as in the Matrix returned by NeuralNetwork.forward
	its FloatBuffers change to get the prediction, and if you have backprop turned on then also the weights etc (w dw stepCache)
	of NeuralNetwork are updated by learning which is computed in opencl.
	<br><br>
	Does the work defined in earlier calls that take Matrix params and return Matrix, pass func, updateModelParams func, etc,
	all in 1 call of opencl using OpenclUtil.callOpenclDependnet for lower lag (than if returned to CPU after each opencl kernel).
	*/
	public void doWork(){
		//Each Matrix has 3 DependParam, 1 for each FloatBuffer (w, dw, and stepCache).
		//StepCache is not used in every Matrix (is null until create as all 0s) so maybe only 2 DependParam sometimes.
		SortedSet<DependOp> ops = dependnet();
		SortedSet<DependParam> insSet = DependOp.dependparamsReadBeforeWritten(ops);
		SortedSet<DependParam> writeSet = DependOp.dependparamsWritten(ops);
		SortedMap<DependParam,Mem> inKeyToMem = new TreeMap();
		//Put (Mem)(FSyMem)Matrix.w (and for .dw and in some cases .stepCache) as values in Map<DependParam,Mem> inKeyToMem,
		//but only those whose DependParam is in insSet.
		for(Matrix m : rememberAllMatrixs){
			/* w dw andOr stepCache may be lazy so dont want to trigger them. Would help if Matrix knew its dependparams before creating the FSyMems which contain those,
			and if it does later create them then use the same DependParam... OR if FSyMem is lazy then just create all 3 right away (TODO is it?).
			??? FSyMem is lazy so I'll modify Matrix to instantly create all 3 FSyMems... unless what if I later want
			more FSyMems for other experimental things. Maybe Matrix should have a map of any string to FSyMem and they should be lazy?
			I do plan to experiment with new kinds of neuralnets, but I was thinking of delaying that until port this code to occamsfuncer
			(which may take a long time). A middle step could be for Matrix to have any number of FSyMem named by a string such as the strings "w" "dw" and "stepCache".
			Yes, do that, and whichever of them are not allocated yet here in doWork() are therefore not part of that work, so just check whichever DependParams they have so far.
			*/
			
			/*BUT... will it make code too long which calls matrix to have to store the FSyMem (or just SyMem or Mem?)
			in a var instead of just
			matrix.w[5] = 6.7f;
			Instead you have to
			FloatBuffer w = matrix.buf("w");
			in a loop read or write w;
			I could keep the w dw and stepCache vars as they are but then whenever I added new stuff that got used alot I'd want to change the Matrix class
			which goes against exploring many possible kinds of neuralnet.
			I think I do want it that way...
			TODO replace Matrix with SortedMap<String,FSyMem> or for it to contain only that.
			Did that, but TODO fix the resulting compile errors.
			
			matrix.w[5] = 6.7f;
			for(Matrix
			*/
			
			//for(String key : m.keys){
			for(Map.Entry<String,FSyMem> entry : m.mems.entrySet()){
				FSyMem fm = entry.getValue();
				DependParam someDP = fm.sy; //may be relevant or not, since we're looking in rememberAllMatrixs.
				if(insSet.contains(someDP)){
					inKeyToMem.put(someDP, fm); //trigger lazy FloatBuffer
				}
			}
		}
		
		//outsSet is subset of writeSet, excluding those which are just temp calculations
		//such as this line in GruLayer: Matrix actMix = g.nonlin(fMix, sum0_plus_sum1__plus_Bmix);
		//In CpuGraph, that line creates Matrixs that are garbcoled after the calculation,
		//but in GpuGraph we avoid creating those Matrixs and instead just create DependOps
		//that multiple of refer to reading and writing of DependParams
		//which OpenclUtil.callOpenclDependnet creates CLMems for without creating FloatBuffer or Matrix for.
		//It all happens in a small fraction of a second usually, data from multiple Matrix goes in,
		//temp calculations happen in CLMems, then copy some of it back into those Matrixs.
		
			
		/*NavigableSet<DependParam> outsSet = TODO how to know which is a temp param vs output that needs to be copied back to Matrix?
			Maybe caller of OpenclGraph funcs has to say so? Look at the funcs in OpenclGraph which take Matrix param(s) and return Matrix.
			Maybe there should be a kind of DependOp that copies between Matrix and the Mems returned by callOpenclDependnet?
				Probably not, cuz that would interfere with callOpenclDependnet statelessly returning Map<DependParam,Mem>.
			Start by finding where the info is created (of if something should be written to a matrix after the DependOps or not),
				considering theres 3 List<DependOp> in OpenclGraph: forwardpropOps, backpropOpsInReverseOrder, trainingpropOps,
					which ListUtil.cat(forwardpropOps, ListUtil.reverse(backpropOpsInReverseOrder), trainingpropOps)
					is 1 of the many allowed dependnet orders of and dependnet() uses to create DependOps
					which know about eachother based on which DependParams they read and write.
				The main usecase is for the Matrixs in Model.getParameters() (including GruLayer.getParameters())
				to be read and written (not necessarily all of them or all parts of them)
				and anything else is probably a temp calculation.
		Map<DependParam,Mem> outs = OpenclUtil.callOpenclDependnet(ins, ops, outKeys);
		...
		TODO give examples of what matrixs should and should not be in Map outs.
		The best working GRU code as of 2020-10-3 is uitool:mutable.mouseai.experiments.recurrentjavasinglechannelmouseai_todoparallel
		which calls NeuralNetworkHelper.makeGru(int parallelSize, int inputDimension, int hiddenDimension, int hiddenLayers, int[] feedforwardSizes, Unaflop decoderUnit, float initParamsStdDev, Random rng)
		which creates GruLayer(s) and FeedForwardLayer(s).
		In that case, out should include:
		-- All new w and dw and stepCache for all Matrix in any getParameters() (which should all be in the root Model.getParameters()).
		-- output of last FeedForwardLayer.
		XX--??? Should it include outputs of the lower layers (some of which are FeedforwardLayer and GruLayer)? Would need those for backprop,
			unless the backprop is already done in this process (which it can be but might want to do this in parts sometimes like if not enough gpu memory)
			so in that way, why give any outputs. Need the top output for prediction. ????
		Should Matrix have a boolean to say its in a getParameters()? Want that Info. Is that bit equal to Matrix.hasStepCacheYet()?
		But how how to know which Matrix is the last output?
		PARTIAL SOLUTION: Use OpenclGraph.isTemp(Matrix) and .setIsTemp(Matrix,boolean).
		*/
		
		Predicate<DependParam> isTempFunc = this::isTemp;
		SortedSet<DependParam> outKeys = setAnd(writeSet, isTempFunc.negate());
		
		Map<DependParam,Mem> openclOuts = OpenclUtil.callOpenclDependnet(inKeyToMem, ops, outKeys);
		
		//FIXME call OpenclGraph.setIsTemp in add, mul, etc funcs???
		
		
		/*copy from openclOuts values (which are Mems) back into Matrixs which each have 3 (lazyEvaled) Mems.
		Not every Mem in openclOuts is for a Model.getParameters() Matrix. It might be the output of the whole NeuralNetwork
		which !isTemp cuz want that for prediction. But it is still in some Matrix, such as returned by LinearLayer.forward,
		so just put it back in whatever Matrix it came from, and caller can get it from there.
		*/
		
		//TODO copy all Mem in openclOuts to whatever Matrix they came from.
		//FIXME need to remember where to put them, maybe with a SortedMap<DependParam,Consumer<Mem>>, but TODO where do the Consumers come from.
		
		//FIXME TODO writeMemsToDependparamsInMatrixs(rememberAllMatrixs, openclOuts);
		
		
		
		
		/*
		//forwardpropOps, backpropOpsInReverseOrder, and trainingpropOps, in any order and some parallel thats allowed by dependnet
		Set<DependOp> ops = dependnet();
		Set<DependParam> outKeys = TODO/*; only those in matrixs that are in Model.getParameters(). The others are middle calculations.
		Maybe Matrix or FSyMem (or Mem) should have a bit that says if its a temp/middle calculation
		vs should be remembered (like input or output in Model.getParameters())?
		Map<DependParam,Mem>outs = OpenclUtil.callOpenclDependnet(ins, ops, outKeys);
		*/
		
		
	}
	
	/** remembers all their Consumer<DependParam> 
	public void rememberMatrixSoCanWriteLater(Matrix m){
	}*/
	public void rememberMatrixsSoCanWriteLater(Matrix... m){
		rememberAllMatrixs.addAll(Arrays.asList(m));
	}
	
	//public static void doDependops(Set<DependOps>)
	
	/**  Immutable. Creates a new dependnet if ops have been added since the last call, else returns from cache. 
	In Trainer.java, for example, backprop and (TODO) training ops, are added or not depending on applyTraining var,
	so if this is only being used for forwardprop/prediction, backpropOpsInReverseOrder and trainingpropOps are empty.
	<br><br>
	Running the output of this in OpenclUtil.callOpenclDependnet will give you the updated FloatBuffers to use in
	the matrixs in a Neuralnet such as a recurrent GruLayer with 5 FeedForwardLayers above it.
	In theory, OpenclUtil.callOpenclDependnet will do this many times faster than the recurrentjava code which only used cpu
	and still fast enough to use in realtime, a few times slower than gaming-low-lag for learning,
	depending how many time cycles (such as 200) it learns sequences (DataSeq) of (such as 200 mouse positions over a few seconds),
	and prediction will be many times faster than gaming-low-lag cuz only need to compute 1 time step forward
	or chaostime (such as 1/3 second, in units of time slices) steps forward.
	Or you might use this for other things. Its a very general way to use OpenclUtil.callOpenclDependnet for
	autodiff backprop and training, but OpenclUtil.callOpenclDependnet itself is more general.
	*/
	public SortedSet<DependOp> dependnet(){
		if(cacheLastDependnetReturned == null) cacheLastDependnetReturned = DependnetBuilder.listToDependnet(
			ListUtil.cat(forwardpropOps, ListUtil.reverse(backpropOpsInReverseOrder), trainingpropOps));
		return cacheLastDependnetReturned;
	}
	
	/*
	//DependnetBuilder forwardpropBuilder = new DependnetBuilder();
	use DependnetBuilder.listToDependnet on concat of the 3 lists of DependOp.
	
	FIXME forwardprop and backpropToDoInReverseOrder need to happen in opposite orders but
	share some CLMems, so need to put them all in 1 DependnetBuilder but not do that until its all done,
	so it seems needs another level of lazyeval of Matrix, since forexample Graph.mul(...)
	must return a Matrix.
	Is there a way to do it without that extra lazyeval?
	...
	Consider creating Runnable objects that will add list of <stringcode,intparallelsize,params[]> to DependnetBuilder
	any time dependnet() is called, and you can keep adding to the multiple lists of <stringcode,intparallelsize,params[]>.
	...
	Consider adding funcs into DependnetBuilder to add another DependnetBuilder's contents,
	BUT that probably wont work cuz the backprop list needs to be added to the other dependnet in the reverse order its created,
	so it has to be the lists (such as forwardprop, backprop, trainingprop) of <stringcode,intparallelsize,params[]>.
			
	/*public List<Task> forwardprop = new ArrayList<>();
	
	//benrayfield made this public for debugging. TODO put DependnetOps in here.
	public List<Task> backpropToDoInReverseOrder = new ArrayList<>();
	
	/** tasks to do for training after forwardprop and backprop, if any *
	public List<Task> trainprop = new ArrayList<>();
	*
	
	FIXME use DependnetBuilder instead of making DependOps directly.
	*/
	
	
	/*In the funcs that take Matrix params and return Matrix and add DependOps to the *prop lists
	(to do all at once in opencl for much lower lag),
	how to know which DependOps each DependOp should wait on?
	At first I wanted to just wait on whichever of them writes the (CLMem)Matrix..stuff..w.mem
	of each of the param Matrixs,
	BUT some Matrix will be written multiple times
	such as summing from multiple Matrixs into Matrix.dw in backprop.
	Should I use a WeakHashMap<Mem,DependOp> for the last DependOp added which writes that?
		Sounds right, but TODO verify.
		Is more dependnetLocking needed similar to what I was planning for another system
			where each Mem alternates being locked for READ vs being locked for WRITE
			and each op would read from some and write to others?
			But in this system, the same op can both read and write the same memory.
		There is a problem: Z reads X. Y writes X. If an order isnt specified between Z and Y,
			then it could be X Y Z or X Z Y, which would have different results.
			So I do need to enforce that at any one time, either
				(1) all DependOps that refer to X are reading X, OR
				(2) at most 1 DependOp is writing X.
			Maybe there should be 3 lockStates per mem: read, readWrite, write.
				That sounds right, but how do I define that in terms of a DependOp forest?
	Should DependOp have pair<lockState,DependParam> instead of just DependParam?
		I'll create a class for that just in case I want to use it: LockPar.
	*/
	
	//TODO use OpenclUtil.callOpenclDependnet here or in a subclass of Graph
	
	protected void onChange(){
		cacheLastDependnetReturned = null;
	}
	
	static final String n = "\n";
	
	public Matrix mul(final Matrix m1, final Matrix m2){
		rememberMatrixsSoCanWriteLater(m1,m2);
		/*Create a Matrix with a new DependParam,
		and add to the task lists in this Graph,
		and somewhere it needs to schedule opencl to copy from CLMem into the
		FloatBuffers in Matrixs.
		*/
		
		final int b = m1.rows;
		final int c = m1.cols;
		final int d = m2.cols;
		//final Matrix out = new Matrix(lazy, m1rows, m2cols);
		final Matrix out = new Matrix(b, d);
		final int outcols = d;
		
		/*public static final String matmulCode1dAs2d =
		"kernel void "+OpenclUtil.newKernelName()+"(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	int bd = get_global_id(0);\r\n"+
		"		const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
		"		const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		float sum = 0;\r\n"+
		"		for(int c=0; c<cSize; c++){\r\n"+
		"			sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		}\r\n"+
		"		bdOut[bd] = sum;\r\n"+
		"}";
		*/
		
		Matrix mOut = new Matrix(b, d);
		
		DependParam bSize = new DependParam("bSize", b);
		DependParam cSize = new DependParam("cSize", c);
		DependParam dSize = new DependParam("dSize", d);
		
		DependOp forward = new DependOp(
				
			/* UPDATE: No dependencies here cuz calculated in dependnet() from forwardpropOps, backpropOpsInReverseOrder, and trainingpropOps.
			new DependOp[]{ FIXME depends on which things that fill the arrays in the 2 param Matrixs. Use LockPar. },
				Think of some examples that could break whatever design I'm considering about this.
				Whatever order of the calls of Graph.java (and this subclass of it),
				there is a calculation done by the DependOps if run in that order,
				and the dependnet must be defined (this DependOp[] here is 1 node's direct dependencies) as
				every possible order which does the same calculation.
				I only have to think about depth 2 outward, such as 2 DependOps that touch the same DependParam,
				one of them reading and one of them readWriting, to make sure that even though that DependParam is
				finished being written by things earlier than those 2 DependOps,
				those 2 DependOps cant run at the same time since it must be readMany_or_writeOne_or_nothing at each
				DependParam at each moment. Find a way to convert List<DependOp> to a dependnet.
				Also upgrade OpenclUtil.callOpenclDependnet to give the dependnet to opencl in a way that
				opencl can do multiple openclNdrangeKernels in parallel (maybe using multiple CLQueue etc)
				instead of computing a single order of the DependOps to do all in 1 CLQueue.
				But first, find a way to convert List<DependOp> to a dependnet, and use it to convert that List
				to a dependnet so openclNdrangeKernels can run it.
				Then, after gru neuralnet is working, add 2 new funcs to OpenclGraph (and Graph)
				which do an array of grunodes or array of lstmnodes (1 func for each) in a single openclNdrangeKernel
				to reduce lag compared to a kernel for multiply, a kernel for plus, etc.
				For each DependParam, alternate ReadLock ReadWrite_or_Write_Lock,
				and use those as part of the dependnet, where DependOps have those as childs,
				BUT... some of those must have DependOps as childs,
				like if dependOpD has readWriteLock on paramXAtItsCycle4 (read=1 rw=2 read=3 rw=4)
				then read=5 is paramX's 5th form and it must have dependOpD as its child.
				I'm unsure if theres an obvious best way to do this or if I'll have to
				have it jiggle the possible dependnets around until it finds something
				between the best possible solution and the List of 1 DependOp at a time.
				Either way Opencl will run faster, that is at least on computers that
				can do multiple kernels at once.
				...
				FIXME, not a complete design yet: Planned solution:
				Each DependParam is used in a linkedlist that alternates Read vs WriteAndOptionallyRead, each wrapped in a DependOp.
				DependParam x = ...;
				DependOp firstWrite = ...(x);
				DependOp firstRead = firstWrite.next();
				DependOp secondWrite = firstRead.next();
				DependOps such as matmul would have at most 1 of those for each DependParam as its childs.
				FIXME which of those (such as firstRead or secondWrite) should depend on a matmul?
				FIXME How to prevent depending on an earlier Read after its written, since these represent state changes of Mem?
				...
				Planned solution:
				create a new class, as a middle layer between OpenclGraph and OpenclUtil.callOpenclDependnet.
				That class will have a WeakHashMap<DependParam,LockState> and implement readMANY and WriteAndOptionallyReadONE,
				in any sequence of adding some anotherNewClass that creates a DependOp but doesnt know its dependencies yet,
				only knows a list of Lockpar (LockState and DependParam) and a code string.
				When one of those (that creates a DependOp) is added, it looks up all its DependParams in that WeakHashMap,
				and for each, if its already READ and the new is READ then it depends on that READ,
				or if its already READ and the new is a WRITEANDOPTIONALLYREAD then add a WRITEANDOPTIONALLYREAD that depends on
				all current readers, or if its already WRITEANDOPTIONALLYREAD then add a READ which depends on that WRITEANDOPTIONALLYREAD
				then depend on that READ. Update the WeakHashMap whenever a DependOp's LockState changes.
				Every DependParam starts as LockState.noLock, so it can go next to READ or WRITEANDOPTIONALLYREAD.
				WRITEANDOPTIONALLYREAD means either of LockState.readWriteLock or LockState.writeLock.
				This will generate 
			*/
				
			"openclNdrangeKernel:kernel void mulForward(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){"+n+
			"	int bd = get_global_id(0);"+n+
			"	const int b = bd/dSize;"+n+
			"	const int d = bd%dSize;"+n+
			"	float sum = 0;"+n+
			"	for(int c=0; c<cSize; c++){"+n+
			"		sum += bc[b*cSize+c]*cd[c*dSize+d];"+n+
			"	}"+n+
			"	bdOut[bd] = sum;"+n+
			"}",
			new ParallelSize(b*d),
			readLock(bSize),
			readLock(cSize),
			readLock(dSize),
			readLock(m1.mem("w").sy),
			readLock(m2.mem("w").sy),
			writeLock(mOut.mem("w").sy)
		);
		
		forwardpropOps.add(forward);
		
		/*Runnable forward = ()->{
			
			String code = "openclNdrangeKernel:mul"+
			
			
			DO IN OPENCL. Add DependOps like in TestOpencl.
			//for (int i = 0; i < m1rows; i++) {
			//	int m1col = m1cols*i;
			//	for (int j = 0; j < m2cols; j++) {
			//		float dot = 0;
			//		for (int k = 0; k < m1cols; k++) {
			//			dot +=  m1.w.get(m1col + k) * m2.w.get(m2cols*k + j);
			//		}
			//		out.w.put(outcols*i + j, dot);
			//	}
			//}
		}*/
		
		if (this.applyBackprop()){
			
			/* final int b = m1.rows;
			final int c = m1.cols;
			final int d = m2.cols;
			*/
			
			/*
			Runnable bp = new Runnable(){
				public void run(){
					
					
					DO IN OPENCL. Add DependOps like in TestOpencl.
					
					b: for (int i = 0; i < m1.rows; i++) {
						int outcol = outcols*i;
						d: for (int j = 0; j < m2.cols; j++) {
							float b = out.dw.get(outcol + j);
							c: for (int k = 0; k < m1.cols; k++) {
								m1.dw.putPlus(c*i+k, m2.w.get(d*k + j) * b);
								m2.dw.putPlus(d*k + j, m1.w.get(c*i + k) * b);
							}
						}
					}
				}
			};
			No, add DependNet to backpropOpsInReverseOrder backpropToDoInReverseOrder.add(new RunnableTask(bp));
			*/
			
			DependOp backprop = new DependOp(
					"openclNdrangeKernel:kernel void mulBackprop(int const bSize, int const cSize, int const dSize,"+n+
					"		global const float* bc, global const float* cd, global const float* bd, global float* bcDeriv, global float* cdDeriv){"+n+
					"	int bdIndex = get_global_id(0);"+n+
					"	const int b = bdIndex/dSize;"+n+
					"	const int d = bdIndex%dSize;"+n+
					"	float wasOut = bd[bdIndex];"+n+
					"	for(int c=0; c<cSize; c++){"+n+
					"		const int bcIndex = b*cSize+c;"+n+
					"		const int cdIndex = c*dSize+d;"+n+
					"		bcDeriv[bcIndex] += cd[cdIndex]*wasOut;"+n+
					"		cdDeriv[cdIndex] += bc[bcIndex]*wasOut;"+n+
					"	}"+n+
					"}",
					new ParallelSize(b*d), //TODO optimize by using 2d and maybe the 32x32x32 opencl group optimization
					readLock(bSize),
					readLock(cSize),
					readLock(dSize),
					readLock(m1.mem("w").sy),
					readLock(m2.mem("w").sy),
					readLock(mOut.mem("w").sy),
					readWriteLock(m1.mem("dw").sy),
					readWriteLock(m2.mem("dw").sy)
			);
			backpropOpsInReverseOrder.add(backprop);
		}
		
		rememberMatrixsSoCanWriteLater(out);
		onChange();
		return out;
	}
	
	public DependOp arraycopy(DependParam from, int fromIndex, DependParam to, int toIndex, int size){
		//TODO optimize there might be opencl command for this faster than a kernel. But kernel will work too.
		return new DependOp(
			"openclNdrangeKernel:kernel void arraycopy(global const float* from, global const int fromIndex, global float* to, global const int toIndex){"+n+
			"	int i = get_global_id(0);"+n+
			"	to[i+toIndex] = from[i+fromIndex];"+n+
			"}",
			new ParallelSize(size),
			readLock(from),
			readLock(new DependParam("fromIndex",fromIndex)),
			writeLock(to), //FIXME should this be readWriteLock since it maybe leaves some indexs as they are?
			readLock(new DependParam("toIndex",toIndex))
		);
	}

	public Matrix concatVectors(Matrix a, Matrix b){
		rememberMatrixsSoCanWriteLater(a,b);
		int aSize = a.rows*a.cols, bSize = b.rows*b.cols;
		//use 2 calls of arraycopy(DependParam ...). To be safe, do one after the other, since using the same CLMem or FloatBuffer 2 times at once might have thread errors?
		forwardpropOps.add(arraycopy(a.mem("w").sy, 0, b.mem("w").sy, 0, aSize));
		forwardpropOps.add(arraycopy(a.mem("dw").sy, 0, b.mem("dw").sy, 0, aSize));
		forwardpropOps.add(arraycopy(b.mem("w").sy, aSize, b.mem("w").sy, 0, bSize));
		forwardpropOps.add(arraycopy(b.mem("dw").sy, aSize, b.mem("dw").sy, 0, bSize));
		if(applyBackprop){
			//FIXME TODO
		}
		
		/*if (m1.cols > 1 || m2.cols > 1) throw new Error("Expected column vectors");
		final Matrix out = new Matrix(m1.rows + m2.rows);
		int loc = 0;
		for (int i = 0; i < m1.w.size; i++) {
			out.w.put(loc, m1.w.get(i));
			out.dw.put(loc, m1.dw.get(i));
			//FIXME? out.stepCache[loc] = m1.stepCache[i];
			loc++;
		}
		for (int i = 0; i < m2.w.size; i++) {
			out.w.put(loc, m2.w.get(i));
			out.dw.put(loc, m2.dw.get(i));
			//FIXME? out.stepCache[loc] = m2.stepCache[i];
			loc++;
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int loc = 0;
					for (int i = 0; i < m1.w.size; i++) {
						m1.w.put(i, out.w.get(loc));
						m1.dw.put(i, out.dw.get(loc));
						//FIXME? m1.stepCache[i] = out.stepCache[loc];
						loc++;
					}
					for (int i = 0; i < m2.w.size; i++) {
						m2.w.put(i, out.w.get(loc));
						m2.dw.put(i, out.dw.get(loc));
						//FIXME? m2.stepCache[i] = out.stepCache[loc];
						loc++;
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
		
		//FIXME TODO rememberMatrixsSoCanWriteLater(out);
		throw new RuntimeException("TODO");
	}

	public Matrix nonlin(Unaflop neuron, Matrix m){
		//rememberMatrixsSoCanWriteLater(m);
		throw new Error("TODO");
		//rememberMatrixsSoCanWriteLater(out);
	}

	public Matrix add(Matrix m1, Matrix m2){
		
		//FIXME call OpenclGraph.setIsTemp in add, mul, etc funcs???
		
		rememberMatrixsSoCanWriteLater(m1,m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) throw new Error("matrix dimension mismatch");
		int size = m1.rows*m1.cols;
		final Matrix out = new Matrix(m1.rows, m1.cols);
		forwardpropOps.add(new DependOp(
			"openclNdrangeKernel:kernel void addForward(global const float* inA, global const float* inB, global float* out){"+n+
			"	int i = get_global_id(0);"+n+
			"	out[i] = inA[i]+inB[i];"+n+
			"}",
			new ParallelSize(size),
			readLock(m1.mem("w").sy),
			readLock(m2.mem("w").sy),
			writeLock(out.mem("w").sy)
		));
		if(this.applyBackprop){
			backpropOpsInReverseOrder.add(new DependOp(
				"openclNdrangeKernel:kernel void addBackprop(global float* inADeriv, global float* inBDeriv, global const float* outDeriv){"+n+
				"	int i = get_global_id(0);"+n+
				"	float o = outDeriv[i];"+n+
				"	inADeriv[i] += o;"+n+ //FIXME should this be o/2?
				"	inBDeriv[i] += o;"+n+
				"}",
				new ParallelSize(size),
				readWriteLock(m1.sy("dw")),
				readWriteLock(m2.sy("dw")),
				readLock(out.sy("dw"))
			));
		}
		rememberMatrixsSoCanWriteLater(out);
		
		/*//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}
		//final Matrix out = new Matrix(lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(m1.rows, m1.cols);
		for (int i = 0; i < m1.w.size; i++) {
			out.w.put(i, m1.w.get(i) + m2.w.get(i));
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.w.size; i++) {
						m1.dw.putPlus(i, out.dw.get(i));
						m2.dw.putPlus(i, out.dw.get(i));
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
		
		return out;
	}
	
	public Matrix elmul(Matrix m1, Matrix m2){
		rememberMatrixsSoCanWriteLater(m1,m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) throw new Error("matrix dimension mismatch");
		int size = m1.rows*m1.cols;
		final Matrix out = new Matrix(m1.rows, m1.cols);
		forwardpropOps.add(new DependOp(
			"openclNdrangeKernel:kernel void elmulForward(global const float* inA, global const float* inB, global float* out){"+n+
			"	int i = get_global_id(0);"+n+
			"	out[i] = inA[i]*inB[i];"+n+
			"}",
			new ParallelSize(size),
			readLock(m1.sy("w")),
			readLock(m2.sy("w")),
			writeLock(out.sy("w")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
		));
		if(this.applyBackprop){
			backpropOpsInReverseOrder.add(new DependOp(
				"openclNdrangeKernel:kernel void elmulBackprop("
				+"		global const float* inA, global const float* inB, global float* inADeriv, global float* inBDeriv, global const float* outDeriv){"+n+
				"	int i = get_global_id(0);"+n+
				"	float o = outDeriv[i];"+n+
				"	inADeriv[i] += inB[i]*o;"+n+
				"	inBDeriv[i] += inA[i]*o;"+n+
				"}",
				new ParallelSize(size),
				readLock(m1.sy("w")),
				readLock(m2.sy("w")),
				readWriteLock(m1.sy("dw")),
				readWriteLock(m2.sy("dw")),
				readLock(out.sy("dw")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
			));
		}
		rememberMatrixsSoCanWriteLater(out);
		return out;
		
		/*if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}
		//final Matrix out = new Matrix(lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(m1.rows, m1.cols);
		for (int i = 0; i < m1.w.size; i++) {
			out.w.put(i, m1.w.get(i) * m2.w.get(i));
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.w.size; i++) {
						m1.dw.putPlus(i, m2.w.get(i) * out.dw.get(i));
						m2.dw.putPlus(i, m1.w.get(i) * out.dw.get(i));
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
	}

	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult){
		//rememberMatrixsSoCanWriteLater(add,rowsOneCol);
		throw new Error("TODO");
		//rememberMatrixsSoCanWriteLater(out);
	}

	public Matrix elmult_rowsCols_to_rowsColsWithColmult(Matrix rowsCols, Matrix rowsOneCol, int colMult){
		//rememberMatrixsSoCanWriteLater(rowsCols,rowsOneCol);
		throw new Error("TODO");
		//rememberMatrixsSoCanWriteLater(out);
	}

	public Matrix oneMinus(Matrix m){
		rememberMatrixsSoCanWriteLater(m);
		int size = m.rows*m.cols;
		final Matrix out = new Matrix(m.rows, m.cols);
		forwardpropOps.add(new DependOp(
			"openclNdrangeKernel:kernel void oneMinusForward(global const float* in, global float* out){"+n+
			"	int i = get_global_id(0);"+n+
			"	out[i] = 1-in[i];"+n+
			"}",
			new ParallelSize(size),
			readLock(m.sy("w")),
			writeLock(out.sy("w")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
		));
		if(this.applyBackprop){
			backpropOpsInReverseOrder.add(new DependOp(
				"openclNdrangeKernel:kernel void oneMinusBackprop(global float* inDeriv, global const float* outDeriv){"+n+
				"	int i = get_global_id(0);"+n+
				"	float o = outDeriv[i];"+n+
				"	inDeriv[i] -= o;"+n+ //FIXME should this be o/2?
				"}",
				new ParallelSize(size),
				readWriteLock(m.sy("dw")),
				readLock(out.sy("dw")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
			));
		}
		rememberMatrixsSoCanWriteLater(out);
		return out;
	}

	public Matrix sub(Matrix m1, Matrix m2){
		rememberMatrixsSoCanWriteLater(m1,m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) throw new Error("matrix dimension mismatch");
		int size = m1.rows*m1.cols;
		final Matrix out = new Matrix(m1.rows, m1.cols);
		forwardpropOps.add(new DependOp(
			"openclNdrangeKernel:kernel void subForward(global const float* inA, global const float* inB, global float* out){"+n+
			"	int i = get_global_id(0);"+n+
			"	out[i] = inA[i]-inB[i];"+n+
			"}",
			new ParallelSize(size),
			readLock(m1.sy("w")),
			readLock(m2.sy("w")),
			writeLock(out.sy("w")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
		));
		if(this.applyBackprop){
			backpropOpsInReverseOrder.add(new DependOp(
				"openclNdrangeKernel:kernel void subBackprop(global float* inADeriv, global float* inBDeriv, global const float* outDeriv){"+n+
				"	int i = get_global_id(0);"+n+
				"	float o = outDeriv[i];"+n+
				"	inADeriv[i] += o;"+n+ //FIXME should this be o/2?
				"	inBDeriv[i] -= o;"+n+
				"}",
				new ParallelSize(size),
				readWriteLock(m1.sy("dw")),
				readWriteLock(m2.sy("dw")),
				readLock(out.sy("dw")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
			));
		}
		rememberMatrixsSoCanWriteLater(out);
		return out;
	}

	public Matrix smul(Matrix m, float s){
		rememberMatrixsSoCanWriteLater(m);
		int size = m.rows*m.cols;
		DependParam inB = new DependParam("inB", s);
		final Matrix out = new Matrix(m.rows, m.cols);
		forwardpropOps.add(new DependOp(
			"openclNdrangeKernel:kernel void elmulForward(global const float* inA, global const float inB, global float* out){"+n+
			"	int i = get_global_id(0);"+n+
			"	out[i] = inA[i]*inB;"+n+
			"}",
			new ParallelSize(size),
			readLock(m.sy("w")),
			readLock(inB),
			writeLock(out.sy("w")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
		));
		if(this.applyBackprop){
			backpropOpsInReverseOrder.add(new DependOp(
				"openclNdrangeKernel:kernel void elmulBackprop("
				+"		global const float* inA, global const float inB, global float* inADeriv, global const float* outDeriv){"+n+
				"	int i = get_global_id(0);"+n+
				"	inADeriv[i] += inB*outDeriv[i];"+n+
				"}",
				new ParallelSize(size),
				readLock(m.sy("w")),
				readLock(inB),
				readWriteLock(m.sy("dw")),
				readLock(out.sy("dw")) //FIXME how will caller of OpenclUtil.callOpenclDependnet know to find Matrix Out using out.sy("w") to copy it into out.w after opencl returns?
			));
		}
		rememberMatrixsSoCanWriteLater(out);
		return out;
	}

	public Matrix neg(Matrix m){
		Matrix out = smul(m,-1f);
		rememberMatrixsSoCanWriteLater(m,out);
		return out;
	}
	
	public Matrix[] acyclicFlow(AcyclicFlowF af, Matrix... ins){
		//rememberMatrixsSoCanWriteLater(ins);
		//String clKernelCode = TODO make it from AcyclicFlowF;
		
		
		//rememberMatrixsSoCanWriteLater(the multiple returned Matrixs);
		throw new Error("FIXME (UPDATE: use acyclicFlow func to replace most of the funcs in Graph) add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in a single opencl ndrange kernel instead of a kernel for every add, multiply, etc, cuz the number of sequential kernels is a bottleneck.");
	}
	
	public void pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting) {
		
		//FIXME call rememberMatrixsSoCanWriteLater(...) on all Matrixs created here
		
		float numerLoss = 0;
		float denomLoss = 0;
		
		int countSteps = sequences.get(0).steps.size();
		
		//benrayfield added param stateResetter so can start at random state to reduce overfitting model.resetState();
		
		
		//FIXME do as DependOp: stateResetter.accept(model);
		
		
		//Graph g = new CpuGraph(applyTraining); //Graph is this.
		int inputSizePerStep = sequences.get(0).steps.get(0).input.cols; //FIXME row vs col?
		int outputSizePerStep = sequences.get(0).steps.get(0).targetOutput.cols; //FIXME row vs col?
		
		//for (DataSequence seq : sequences) {
		for (int stepNum=0; stepNum<countSteps; stepNum++){
			//these are steps all at the same time.
			//The earlier and later steps are looped around this (TODO)
			Matrix inputOfAllSteps = new Matrix(inputSizePerStep, sequences.size()); //FIXME rows vs cols backward?
			Matrix correctOutputOfAllSteps = new Matrix(outputSizePerStep, sequences.size()); //FIXME rows vs cols backward?
			for(int seqNum=0; seqNum<sequences.size(); seqNum++){
				DataStep step = sequences.get(seqNum).steps.get(stepNum);
				//FIXME since I'm using only 1 input and 1 output, might have got rows vs cols backward
				
				/*
				//System.arraycopy(
				FIXME do this as the arraycopy func that returns a DependOp: FMem.arraycopy(
					step.input.w.mem(), 0, //copy from (all)
					inputOfAllSteps.w.mem(), seqNum*inputSizePerStep, //copy to (range)
					inputSizePerStep);
				//System.arraycopy(
				FIXME do this as the arraycopy func that returns a DependOp: FMem.arraycopy(
					step.targetOutput.w.mem(), 0, //copy from (all)
					correctOutputOfAllSteps.w.mem(), seqNum*outputSizePerStep, //copy to (range)
					outputSizePerStep);
				*/
				
				//FIXME is forwardpropOps the right list?
				forwardpropOps.add(arraycopy(
					step.input.sy("w"), 0, //copy from (all)
					inputOfAllSteps.sy("w"), seqNum*inputSizePerStep, //copy to (range)
					inputSizePerStep));
				forwardpropOps.add(arraycopy(
					step.targetOutput.sy("w"), 0, //copy from (all)
					correctOutputOfAllSteps.sy("w"), seqNum*outputSizePerStep, //copy to (range)
					outputSizePerStep));
				
			}
			
			//FIXME do as DependOp: Matrix output = model.forward(inputOfAllSteps, g);
			
			//if(output.rows != correctOutputOfAllSteps.rows || output.cols != correctOutputOfAllSteps.cols)
			//	throw new Error("output and correctOutputOfAllSteps are diff sizes");
			//Matrix output = model.forward(step.input, g);
			
			//FIXME cant do this during opencl call: outputListener.accept(output); //benrayfield added this to avoid recomputing it in UnidimView
			
			
			//FIXME cant do this during opencl call: float loss = lossReporting.measure(output, correctOutputOfAllSteps);
			
			float loss = 1000; //FIXME dont make this up. it was working before started adding opencl, in recurrentjava autodiff, so go back there to figure out what this should be.
			
			//benrayfield: System.out.println("pass loss="+loss);
			if(Float.isNaN(loss) || Float.isInfinite(loss)) {
				throw new Error("loss is not finite: "+loss);
				//return loss;
			}
			numerLoss += loss;
			denomLoss++;			
			if(applyTraining) {
				//FIXME do as DependOp: lossTraining.backward(output, correctOutputOfAllSteps);
			}
		}
		
		//FIXME ((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
		
		if (applyTraining) {
			updateModelParams(params, model);
		}
		
		//FIXME ((CpuGraph)g).doTasks(); //FIXME are these the right times to doTasks, before andOr after updateModelParams?
		double retStats = numerLoss/denomLoss;
		
		/*FIXME: cant "Do all tasks at once in OpenclUtil.callOpenclDependnet." thats not supposed to happen in
		the funcs that add DependOps. Its supposed to happen after all relevant DependOps have been added.
		*
		
		//return ret;
		
		FIXME return retStats where? Could return it into a Mem size 1 float.
		*/
	}
	
	/** the Set<Matrix> is just to look for the DependParams in. Only those with the DependParams that are in keys of SortedMap<DependParam,Mem> are written,
	of the 3 DependParams (w dw stepCache) per Matrix, which any or all of may be written.
	*/
	public static void writeMemsToDependparamsInMatrixs(Set<Matrix> search, SortedMap<DependParam,Mem> writes){
		//FIXME TODO TODO
	}
	
	//FIXME
	/*First I'm copying code from Trainer.pass and things it calls, to here, then upgrade to use DependOps.
	Then make sure it all happens in 1 call of OpenclUtil.callOpenclDependnet (maybe 300 kernels) for lower lag,
	which reads from Matrixs, does the many opencl kernels, then writes to Matrixs
	*/

	public void updateModelParams(RnnParams p, Model model){
		//FIXME call rememberMatrixsSoCanWriteLater(...) on all Matrixs created here
		
		for (Matrix m : model.getParameters()){
			
			trainingpropOps.add(new DependOp(
				"openclNdrangeKernel:kernel void updateModelParam(global const float* w, global float* dw, global float* stepCache,"+
				"		float const rjTrainerDecayRate, float const rjTrainerGradientClipValue, float const learnRate, float const rjTrainerSmoothEpsilon, float const rjTrainerRegularization){"+n+
				"	int i = get_global_id(0);"+n+
				"	float mwdi = dw[i];"+n+
				"	stepCache[i] = stepCache[i]*rjTrainerDecayRate+(1-rjTrainerDecayRate)*mwdi*mwdi;"+n+
				"	mwdi = max(-rjTrainerGradientClipValue, min(mwdi, rjTrainerGradientClipValue)); //gradient clip. FIXME why is this after its first used?"+n+
				"	w[i] -= learnRate * mdwi / Math.sqrt(stepCache[i] + rjTrainerSmoothEpsilon) - rjTrainerRegularization * w[i];"+n+
				"	dw[i] = 0;"+n+
				"}",
				new ParallelSize(m.cols*m.rows),
				readLock(m.sy("w")),
				readWriteLock(m.sy("dw")),
				//FIXME TODO readWriteLock(m.stepCache().sy),
				readLock(new DependParam("rjTrainerDecayRate",p.rjTrainerDecayRate)),
				readLock(new DependParam("rjTrainerGradientClipValue",p.rjTrainerGradientClipValue)),
				readLock(new DependParam("learnRate",p.learnRate)),
				readLock(new DependParam("rjTrainerSmoothEpsilon",p.rjTrainerSmoothEpsilon)),
				readLock(new DependParam("rjTrainerRegularization",p.rjTrainerRegularization))
			));
			
			/*
			TODO add DependOps instead of this code copied from Traininer...
			
			FSyMem stepCache = m.stepCache(); //lazyCreate
			for (int i = 0; i < m.w.size; i++) {
				
				// rmsprop adaptive learning rate
				float mdwi = m.dw.get(i);
				//m.stepCache[i] = m.stepCache[i] * p.rjTrainerDecayRate + (1 - p.rjTrainerDecayRate) * mdwi * mdwi;
				stepCache.put(i, stepCache.get(i) * p.rjTrainerDecayRate + (1 - p.rjTrainerDecayRate) * mdwi * mdwi);
				
				// gradient clip
				if (mdwi > p.rjTrainerGradientClipValue) {
					mdwi = p.rjTrainerGradientClipValue;
				}
				if (mdwi < -p.rjTrainerGradientClipValue) {
					mdwi = -p.rjTrainerGradientClipValue;
				}
				
				// update (and regularize)
				//m.w[i] += - p.learnRate * mdwi / Math.sqrt(m.stepCache[i] + p.rjTrainerSmoothEpsilon) - p.rjTrainerRegularization * m.w[i];
				float mwi = m.w.get(i);
				m.w.put(i, (float)(mwi - p.learnRate * mdwi / Math.sqrt(stepCache.get(i) + p.rjTrainerSmoothEpsilon) - p.rjTrainerRegularization * mwi));
				m.dw.put(i,0);
				
				//benrayfield
				//FIXME how can I testDelayedUpdateOfWeights when weightChange is decaying m.w?
				//Also delay the processing of m.dw?
				//I only want to delay until the end of a batch.
				//In the parallel code, the weight arrays are shared and the node states etc
				//are parallelSize times bigger.
				//Since weight arrays are shared, how much is that affecting backprop?
				//Maybe I can just call updateModelParams at end of batch.
				
				
			}
			*/
		}
	}

}
