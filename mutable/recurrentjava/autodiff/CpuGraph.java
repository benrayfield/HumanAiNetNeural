package mutable.recurrentjava.autodiff;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import immutable.acyclicflow.AcyclicFlowF;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.rbm.learnloop.OpenclProgs;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.rnn.RnnParams;
import immutable.util.MathUtil;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.util.task.RunnableTask;
import mutable.util.task.Task;

/** a Graph optimized for CPUs. A Graph is a mutable builder of numberCrunching ops with backprop (and in some cases also training) built in. */
public strictfp class CpuGraph implements Graph{
	
	public boolean isLazy(){ return false; }
	public boolean hasWork(){ return false; }
	public void doWork(){}
	
	//FIXME make this an interface and rename this class to CpuGraph
	
	//TODO use OpenclUtil.callOpenclDependnet here or in a subclass of Graph
	
	/** (benrayfield changed) FIXME as of 2020-1-10 this var isnt used yet
	and I was considering making 2 subclasses of Graph,
	and maybe I still should?
	If false, does the original RecurrentJava behaviors
	(slightly modified by BenRayfield).
	If true, delays those to do in LWJGL openCL all at once
	in a single opencl call before returning to java for low lag.
	*
	protected boolean opencl;
	*/
	
	//benrayfield made this protected
	protected boolean applyBackprop;
	public boolean applyBackprop(){ return applyBackprop; }
	
	/** tasks to do before backprop, if any.
	benrayfield added this to put DependnetOps in,
	the parts that normally happen as soon as a Matrix is created will instead
	be lazyEvaled all at once in opencl, or when opencl is not used then still instant.
	*/
	public List<Task> forwardprop = new ArrayList<>();
	
	//benrayfield made this public for debugging. TODO put DependnetOps in here.
	public List<Task> backpropToDoInReverseOrder = new ArrayList<>();
	
	/** tasks to do for training after forwardprop and backprop, if any */
	public List<Task> trainprop = new ArrayList<>();
	
	public CpuGraph() {
		this.applyBackprop = true;
	}
	
	public CpuGraph(boolean applyBackprop) {
		this.applyBackprop = applyBackprop;
	}
	
	public void doTasks(){
		List<Task> tasks = new ArrayList(forwardprop);
		tasks.addAll(MathUtil.reverse(backpropToDoInReverseOrder));
		tasks.addAll(trainprop);
		if(tasks.stream().allMatch(x->(x instanceof RunnableTask))) {
			Task.doTasksInCpu(tasks);
		}else{
			throw new Error("cant Task.doTasksInOpencl(tasks); cuz redesigning to use CpuGraph and OpenclGraph");
		}
		forwardprop.clear();
		backpropToDoInReverseOrder.clear();
		trainprop.clear();
	}
	
	/** (benrayfield made this func) Was it wrong to copy the stepcache?
	I dont see other funcs accessing stepcache places other than in Trainer.
	Maybe thats cuz this op isnt used except inside neural nodes
	and only the Model.getParameters() (such as weights) use stepCache.
	*/
	public Matrix concatVectors(final Matrix m1, final Matrix m2){
		if (m1.cols > 1 || m2.cols > 1) {
			throw new Error("Expected column vectors");
		}
		final Matrix out = new Matrix(m1.rows + m2.rows);
		int loc = 0;
		/*boolean doStepCache = m1.hasStepCacheYet() || m2.hasStepCacheYet();
		BiMem<FloatBuffer> outStepCache = out.stepCache;
		BiMem<FloatBuffer> outStepCache = 
		if(m1.hasStepCacheYet() || m2.hasStepCacheYet()){
			out.stepCache();
		}
		BiMem<FloatBuffer> outStepCache = out.stepCache;
		*/
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		FSyMem m1w = m1.mem("w");
		FSyMem m1dw = m1.mem("dw");
		FSyMem m2w = m2.mem("w");
		FSyMem m2dw = m2.mem("dw");
		for (int i = 0; i < m1.size; i++) {
			outW.put(loc, m1w.get(i));
			outDw.put(loc, m1dw.get(i));
			//FIXME? out.stepCache[loc] = m1.stepCache[i];
			loc++;
		}
		for (int i = 0; i < m2.size; i++) {
			outW.put(loc, m2w.get(i));
			outDw.put(loc, m2dw.get(i));
			//FIXME? out.stepCache[loc] = m2.stepCache[i];
			loc++;
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int loc = 0;
					for (int i = 0; i < m1.size; i++) {
						m1w.put(i, outW.get(loc));
						m1dw.put(i, outDw.get(loc));
						//FIXME? m1.stepCache[i] = out.stepCache[loc];
						loc++;
					}
					for (int i = 0; i < m2.size; i++) {
						m2w.put(i, outW.get(loc));
						m2dw.put(i, outDw.get(loc));
						//FIXME? m2.stepCache[i] = out.stepCache[loc];
						loc++;
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix nonlin(final Unaflop neuron, final Matrix m){
		//final Matrix out = new Matrix(m.lazy, m.rows, m.cols);
		final Matrix out = new Matrix(m.rows, m.cols);
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		FSyMem m1w = m.mem("w");
		FSyMem m1dw = m.mem("dw");
		final int n = m.size;
		for (int i = 0; i < n; i++) {
			outW.put(i, neuron.forward(m1w.get(i)));
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < n; i++) {
						m1dw.putPlus(i, neuron.deriv(m1w.get(i)) * outDw.get(i));
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	/*public static boolean allLazyOrAllNotLazy(Matrix... m){
		boolean ret = m[0].lazy;
		for(int i=0; i<m.length; i++) if(m[i].lazy != ret) throw new Error("Some lazy and some nonlazy");
		return ret;
	}*/
	
	public Matrix mul(final Matrix m1, final Matrix m2){
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.cols != m2.rows) {
			throw new Error("matrix dimension mismatch");
		}
		
		final int m1rows = m1.rows;
		final int m1cols = m1.cols;
		final int m2cols = m2.cols;
		//final Matrix out = new Matrix(lazy, m1rows, m2cols);
		final Matrix out = new Matrix(m1rows, m2cols);
		
		FSyMem m1w = m1.mem("w");
		FSyMem m1dw = m1.mem("dw");
		FSyMem m2w = m2.mem("w");
		FSyMem m2dw = m2.mem("dw");
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		
		final int outcols = m2cols;
		if(false){
		//if(RjOptions.opencl){
			/** from OpenclProgs.java:
			** given float[b][c] and float[c][d] returns float[b][d] *
			public static synchronized float[][] matmul(float[][] bc, float[][] cd){
				int bSize = bc.length, cSize = bc[0].length, dSize = cd[0].length;
				if(cd.length != cSize) throw new Error("Sizes dont match");
				//FIXME verify sizes match and are rectangle arrays
				float[] bd1d = matmul(bSize, cSize, dSize, OpenclUtil.array2dTo1d(bc), OpenclUtil.array2dTo1d(cd));
				return OpenclUtil.array1dTo2d(bd1d,bSize);
			}
			** bc.length==bSize*cSize && cd.length==cSize*dSize *
			public static synchronized float[] matmul(int bSize, int cSize, int dSize, float[] bc, float[] cd){
				Object[] out = OpenclUtil.callOpencl(
					
					//FIXME slower, try this until get the right answer then start using matmulCode1dAs2d instead and make that work
					matmulCode1dAs2d, new int[]{bSize*dSize},
					
					//FIXME This gets about 3.5 gflops on my 4x1.6GhzLaptop, while the other only about 2. Both give wrong answer,
					//this one gives 0 and other one gives it appears 1 of the input numbers, so I'm going back to the slower 1d one
					//while I fix that then come back to this for speed if I can
					//matmulCode2d, new int[]{bSize, dSize},
					
					bSize, cSize, dSize, bc, cd, new float[bSize*dSize]);
				return (float[]) out[out.length-1];
			}
			public static final String matmulCode1dAs2d =
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
			BUT I NEED DOUBLES INSTEAD OF FLOATS:
			*/
			
			//TODO optimize by skipping the float[] step as OpenclUtil uses FloatBuffer
			float[] bdOut = OpenclProgs.matmul(
				m1rows, //bSize
				m1cols, //cSize
				m2cols, //dSize
				m1.mem("w").toFloatArray(), //MathUtil.toFloats(m1.w), //bc
				m2.mem("w").toFloatArray() //MathUtil.toFloats(m2.w) //cd
			);
			if(bdOut.length != out.size) throw new Error("matmul broke");
			for(int i=0; i<out.size; i++){
				outW.put(i, bdOut[i]);
			}
			
			/* Dont use double in opencl cuz some computers dont support it.
			//OpenclUtil doesnt modify any of the inputs or outputs, even if marked as mutable like "global double* bdOut".
			Object[] openclOut = OpenclUtil.callOpencl(
				OpenclProgs.openclNdrangeCode_matmulDouble,
				new int[]{out.w.length},
				m1rows, //int const bSize
				m1cols, //int const cSize
				m2cols, //int const dSize
				//FIXME are dims backward?
				m1.w, //global const double* bc
				//FIXME are dims backward?
				m2.w, //global const double* cd
				out.w  //global double* bdOut
			);
			//FIXME are dims backward?
			double[] bdOut = (double[]) openclOut[5];
			System.arraycopy(bdOut, 0, out.w, 0, out.w.length); //cuz OpenclUtil doesnt modify params
			*/
			
			
		}else{
			for (int i = 0; i < m1rows; i++) {
				int m1col = m1cols*i;
				for (int j = 0; j < m2cols; j++) {
					float dot = 0;
					for (int k = 0; k < m1cols; k++) {
						dot +=  m1w.get(m1col + k) * m2w.get(m2cols*k + j);
					}
					outW.put(outcols*i + j, dot);
				}
			}
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.rows; i++) {
						int outcol = outcols*i;
						for (int j = 0; j < m2.cols; j++) {
							float b = outDw.get(outcol + j);
							for (int k = 0; k < m1.cols; k++) {
								m1dw.putPlus(m1cols*i+k, m2w.get(m2cols*k + j) * b);
								m2dw.putPlus(m2cols*k + j, m1w.get(m1cols*i + k) * b);
							}
						}
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix add(final Matrix m1, final Matrix m2){
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}

		//final Matrix out = new Matrix(lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(m1.rows, m1.cols);
		
		FSyMem m1w = m1.mem("w");
		FSyMem m1dw = m1.mem("dw");
		FSyMem m2w = m2.mem("w");
		FSyMem m2dw = m2.mem("dw");
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.get(i) + m2w.get(i));
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.size; i++) {
						m1dw.putPlus(i, outDw.get(i));
						m2dw.putPlus(i, outDw.get(i));
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	/** Example add.rows=200 add.cols=5 rowsOneCol.rows=200 rowsOneCol.cols=1 colMult=5 returns rows=200 cols=5.
	Benrayfields upgrading of recurrentjava to opencl is putting multiple cols as parallelSize
	(unsure if it should be rows or cols yet 2019-5-9, probably cols... UPDATE: 2020-10 whatever the code is now, it works),
	and the bias needs to be added to all parallelIndex vecs, unlike matmul which (it appears) already does.
	Copying and modifying the code from add(...).
	Planning to opencl upgrade after the upgrade to parallelSize and parallelIndex vars.
	<br><br>
	FIXME is this the same as add(Matrix add, Matrix concatVectors colMult of them)? And should it be?
	*/
	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult){
		//boolean lazy = allLazyOrAllNotLazy(add, rowsOneCol);
		if(add.rows != rowsOneCol.rows || add.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lazy, add.rows, add.cols);
		final Matrix out = new Matrix(add.rows, add.cols);
		
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		FSyMem addW = add.mem("w");
		FSyMem addDw = add.mem("dw");
		FSyMem rowsOneColW = rowsOneCol.mem("w");
		FSyMem rowsOneColDw = rowsOneCol.mem("dw");
		
		int offset = 0;
		for(int col=0; col<colMult; col++){
			for (int i = 0; i < rowsOneCol.size; i++){
				outW.put(offset, addW.get(offset) + rowsOneColW.get(i));
				offset++;
			}
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int offset = 0;
					for(int col=0; col<colMult; col++){
						for (int i = 0; i < rowsOneColW.size; i++) {
							
							//m1.dw[i] += out.dw[i];
							addDw.putPlus(offset, outDw.get(offset));
							
							//m2.dw[i] += out.dw[i];
							//FIXME this adjusts the bias colMult times more.
							//Should it be that vs just 1 times as much?
							//If did this with concatVectors of bias, which would it do?
							//Could be a problem if the dws arent changed equally?
							rowsOneColDw.putPlus(i, outDw.get(offset));
							
							offset++;
						}
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix elmult_rowsCols_to_rowsColsWithColmult(Matrix rowsCols, Matrix rowsOneCol, int colMult){
		//boolean lazy = allLazyOrAllNotLazy(rowsCols, rowsOneCol);
		if (rowsCols.rows != rowsOneCol.rows || rowsCols.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lazy, rowsCols.rows, rowsCols.cols);
		final Matrix out = new Matrix(rowsCols.rows, rowsCols.cols);
		
		FSyMem rowsColsW = rowsCols.mem("w");
		FSyMem rowsColsDw = rowsCols.mem("dw");
		FSyMem rowsOneColW = rowsOneCol.mem("w");
		FSyMem rowsOneColDw = rowsOneCol.mem("dw");
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		
		int rows = rowsCols.rows;
		int cols = colMult;
		int offset = 0;
		for(int row=0; row<rows; row++){
			for(int col=0; col<rowsCols.cols; col++){
				outW.put(offset, rowsColsW.get(offset) * rowsOneColW.get(row));
				offset++;
			}
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int offset = 0;
					for(int row=0; row<rows; row++){
						for(int col=0; col<rowsCols.cols; col++){
							//out.w[offset] = rowsCols.w[offset] * rowsOneCol.w[row];
							rowsColsDw.putPlus(offset, rowsOneColW.get(row) * outDw.get(offset));
							rowsOneColDw.putPlus(row, rowsColsW.get(offset) * outDw.get(offset));
							offset++;
						}
					}
					/*for (int i = 0; i < m1.w.length; i++) {
						m1.dw[i] += m2.w[i] * out.dw[i];
						m2.dw[i] += m1.w[i] * out.dw[i];
					}*/
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix oneMinus(final Matrix m){
		Matrix ones = Matrix.ones(m.rows, m.cols);
		Matrix out = sub(ones, m);
		return out;
	}
	
	public Matrix sub(final Matrix m1, final Matrix m2){
		Matrix out = add(m1, neg(m2));
		return out;
	}
	
	public Matrix smul(final Matrix m, final float s){
		Matrix m2 = Matrix.uniform(m.rows, m.cols, s);
		Matrix out = elmul(m, m2);
		return out;
	}
		
	public Matrix neg(final Matrix m){
		Matrix negones = Matrix.negones(m.rows, m.cols);
		Matrix out = elmul(negones, m);
		return out;
	}
	
	public Matrix elmul(final Matrix m1, final Matrix m2){
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}
		//final Matrix out = new Matrix(lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(m1.rows, m1.cols);
		
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		FSyMem m1w = m1.mem("w");
		FSyMem m1dw = m1.mem("dw");
		FSyMem m2w = m2.mem("w");
		FSyMem m2dw = m2.mem("dw");
		
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.get(i) * m2w.get(i));
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.size; i++) {
						m1dw.putPlus(i, m2w.get(i) * outDw.get(i));
						m2dw.putPlus(i, m1w.get(i) * outDw.get(i));
					}
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix[] acyclicFlow(AcyclicFlowF af, Matrix... ins){
		throw new Error("TODO");
	}
	
	public void pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting){
		Trainer.pass(params, outputListener, stateResetter, model, sequences, applyTraining, lossTraining, lossReporting);
	}
	
	public void updateModelParams(RnnParams p, Model model){
		Trainer.updateModelParams(p, model);
	}

}
