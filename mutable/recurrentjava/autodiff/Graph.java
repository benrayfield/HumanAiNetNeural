package mutable.recurrentjava.autodiff;
import java.io.Flushable;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import immutable.acyclicflow.AcyclicFlowF;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.rbm.learnloop.OpenclProgs;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.rnn.RnnParams;
import immutable.util.MathUtil;
import mutable.compilers.opencl.BiMem;
import mutable.compilers.opencl.OpenclUtil;
import mutable.recurrentjava.RjOptions;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.model.NeuralNetwork;
import mutable.util.ManualLazy;
import mutable.util.task.RunnableTask;
import mutable.util.task.Task;

/** A Graph is a mutable builder of numberCrunching ops with backprop (and in some cases also training) built in.
BenRayfield found opensourceMITLicensed RecurrentJava code which seemed to do the right calculations but only on CPU,
so he is (as of 2020-5-2) still OpenCL optimizing that whole autodiff system
and plans to add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in
a single opencl ndrange kernel instead of a kernel for every add, multiply, etc,
cuz the number of sequential kernels is a bottleneck.
*/
public interface Graph extends ManualLazy{
	
	public boolean applyBackprop();
	
	//TODO rename this recurrentjava to benrayfieldsrecurrentjavafork or something shorter

	/** (benrayfield made this func) Was it wrong to copy the stepcache?
	I dont see other funcs accessing stepcache places other than in Trainer.
	Maybe thats cuz this op isnt used except inside neural nodes
	and only the Model.getParameters() (such as weights) use stepCache.
	*/
	public Matrix concatVectors(final Matrix m1, final Matrix m2);
	
	//FIXME implement most of the funcs in Graph using acyclicFlow(AcyclicFlow,Matrix...),
	//such as sub(Matrix,Matrix) and oneMinus(Matrix), but I'm unsure if I want every input and output var in
	//AcyclicFlow to be its own Matrix. Maybe specify that in AcyclicFlow class.
		
	public Matrix nonlin(final Unaflop neuron, final Matrix m);
	
	public Matrix mul(final Matrix m1, final Matrix m2);
	
	public Matrix add(final Matrix m1, final Matrix m2);
	
	/** Example add.rows=200 add.cols=5 rowsOneCol.rows=200 rowsOneCol.cols=1 colMult=5 returns rows=200 cols=5.
	Benrayfields upgrading of recurrentjava to opencl is putting multiple cols as parallelSize
	(unsure if it should be rows or cols yet 2019-5-9, probably cols),
	and the bias needs to be added to all parallelIndex vecs, unlike matmul which (it appears) already does.
	Copying and modifying the code from add(...).
	Planning to opencl upgrade after the upgrade to parallelSize and parallelIndex vars.
	<br><br>
	FIXME is this the same as add(Matrix add, Matrix concatVectors colMult of them)? And should it be?
	*/
	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult);
	
	public Matrix elmult_rowsCols_to_rowsColsWithColmult(Matrix rowsCols, Matrix rowsOneCol, int colMult);
	
	public Matrix oneMinus(final Matrix m);
	
	public Matrix sub(final Matrix m1, final Matrix m2);
	
	public Matrix smul(final Matrix m, final float s);
	
	public default Matrix smul(final float s, final Matrix m) {
		return smul(m,s);
	}
	
	public Matrix neg(final Matrix m);
	
	public Matrix elmul(final Matrix m1, final Matrix m2);
	
	/*public default Matrix[] forwardpropGruNodes(Object todoWhatParams){
		throw new Error("FIXME add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in a single opencl ndrange kernel instead of a kernel for every add, multiply, etc, cuz the number of sequential kernels is a bottleneck.");
	}
	
	public default Matrix[] backpropGruNodes(Object todoWhatParams){
		throw new Error("FIXME add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in a single opencl ndrange kernel instead of a kernel for every add, multiply, etc, cuz the number of sequential kernels is a bottleneck.");
	}
	
	public default Matrix[] forwardpropLstmNodes(Object todoWhatParams){
		throw new Error("FIXME add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in a single opencl ndrange kernel instead of a kernel for every add, multiply, etc, cuz the number of sequential kernels is a bottleneck.");
	}
	
	public default Matrix[] backpropLstmNodes(Object todoWhatParams){
		throw new Error("FIXME add a few more funcs to Graph so forward and backprop thru a gru node and lstm node can be done in a single opencl ndrange kernel instead of a kernel for every add, multiply, etc, cuz the number of sequential kernels is a bottleneck.");
	}
	
	TODO a Matrix that does an AcyclicFlow, which forwardpropGruNodes and backpropLstmNodes etc could be implemented as instead of complicating Graph interface with those funcs.
		That would be useful for also being able to optimize new kinds of neuralnets AI thinks of at runtime to try.
		Maybe use Biflop and Unaflop interfaces for this? Or use some of the physicsmata mathevo code? Or some of the acyclicflow int[] code (2 int12 ptrs and an int8 op)?
		That could be the cpu form of it, but still need to generate opencl code. This will be compatible withopencl wallet and spend ops.
	*/
	
	/** For each input in AcyclicFlow, theres a Matrix param. For each output, theres a returned Matrix. */
	public Matrix[] acyclicFlow(AcyclicFlowF af, Matrix... ins);
	
	/** This is a redesign of RecurrentJava Trainer.pass(...).
	Some of the params are used during CpuGraph but only at the start andOr end of GpuGraph.
	<br><br>
	Schedule thinking andOr training of the NeuralNetwork which the earlier calls, such as mul(Matrix,Matrix)
	are part of its forward and backprop steps.
	Gets Matrixs from it. You still have to call run() (as this implements Runnable) to do all those things if this is an OpenclGraph,
	else if its CpuGraph does it here.
	*
	public void train(NeuralNetwork net);
	*/
	public void pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
		Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting);
	
	/** normally called by pass(...) */
	public void updateModelParams(RnnParams p, Model model);
	
}
