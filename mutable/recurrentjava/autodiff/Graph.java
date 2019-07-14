package mutable.recurrentjava.autodiff;
import java.util.ArrayList;
import java.util.List;

import immutable.rbm.learnloop.OpenclProgs;
import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.compilers.opencl.OpenclUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.recurrentjava.RjOptions;
import mutable.recurrentjava.matrix.Matrix;


public class Graph {
	boolean applyBackprop;
	
	//benrayfield made this public for debugging
	public List<Runnable> backprop = new ArrayList<>();
	
	public Graph() {
		this.applyBackprop = true;
	}
	
	public Graph(boolean applyBackprop) {
		this.applyBackprop = applyBackprop;
	}
	
	public void backward() {
		for (int i = backprop.size()-1; i >= 0; i--) {
			backprop.get(i).run();
		}
	}
	
	public Matrix concatVectors(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.cols > 1 || m2.cols > 1) {
			throw new Exception("Expected column vectors");
		}
		final Matrix out = new Matrix(m1.rows + m2.rows);
		int loc = 0;
		for (int i = 0; i < m1.w.length; i++) {
			out.w[loc] = m1.w[i];
			out.dw[loc] = m1.dw[i];
			out.stepCache[loc] = m1.stepCache[i];
			loc++;
		}
		for (int i = 0; i < m2.w.length; i++) {
			out.w[loc] = m2.w[i];
			out.dw[loc] = m2.dw[i];
			out.stepCache[loc] = m2.stepCache[i];
			loc++;
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int loc = 0;
					for (int i = 0; i < m1.w.length; i++) {
						m1.w[i] = out.w[loc];
						m1.dw[i] = out.dw[loc];
						m1.stepCache[i] = out.stepCache[loc];
						loc++;
					}
					for (int i = 0; i < m2.w.length; i++) {
						m2.w[i] = out.w[loc];
						m2.dw[i] = out.dw[loc];
						m2.stepCache[i] = out.stepCache[loc];
						loc++;
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	public Matrix nonlin(final Unaflop neuron, final Matrix m) throws Exception {
		final Matrix out = new Matrix(m.rows, m.cols);
		final int n = m.w.length;
		for (int i = 0; i < n; i++) {
			out.w[i] = neuron.forward(m.w[i]);
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < n; i++) {
						m.dw[i] += neuron.deriv(m.w[i]) * out.dw[i];
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	public Matrix mul(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.cols != m2.rows) {
			throw new Exception("matrix dimension mismatch");
		}
		
		final int m1rows = m1.rows;
		final int m1cols = m1.cols;
		final int m2cols = m2.cols;
		final Matrix out = new Matrix(m1rows, m2cols);
		final int outcols = m2cols;
		if(RjOptions.opencl){
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
			
			float[] bdOut = OpenclProgs.matmul(
				m1rows, //bSize
				m1cols, //cSize
				m2cols, //dSize
				MathUtil.toFloats(m1.w), //bc
				MathUtil.toFloats(m2.w) //cd
			);
			if(bdOut.length != out.w.length) throw new Error("matmul broke");
			for(int i=0; i<out.w.length; i++){
				out.w[i] = bdOut[i];
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
					double dot = 0;
					for (int k = 0; k < m1cols; k++) {
						dot +=  m1.w[m1col + k] * m2.w[m2cols*k + j];
					}
					out.w[outcols*i + j] = dot;
				}
			}
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.rows; i++) {
						int outcol = outcols*i;
						for (int j = 0; j < m2.cols; j++) {
							double b = out.dw[outcol + j];
							for (int k = 0; k < m1.cols; k++) {
								m1.dw[m1cols*i + k] += m2.w[m2cols*k + j] * b;
								m2.dw[m2cols*k + j] += m1.w[m1cols*i + k] * b;
							}
						}
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	public Matrix add(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
		final Matrix out = new Matrix(m1.rows, m1.cols);
		for (int i = 0; i < m1.w.length; i++) {
			out.w[i] = m1.w[i] + m2.w[i];
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.w.length; i++) {
						m1.dw[i] += out.dw[i];
						m2.dw[i] += out.dw[i];
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	/** Example add.rows=200 add.cols=5 rowsOneCol.rows=200 rowsOneCol.cols=1 colMult=5 returns rows=200 cols=5.
	Benrayfields upgrading of recurrentjava to opencl is putting multiple cols as parallelSize
	(unsure if it should be rows or cols yet 2019-5-9, probably cols),
	and the bias needs to be added to all parallelIndex vecs, unlike matmul which (it appears) already does.
	Copying and modifying the code from add(...).
	Planning to opencl upgrade after the upgrade to parallelSize and parallelIndex vars.
	<br><br>
	FIXME is this the same as add(Matrix add, Matrix concatVectors colMult of them)? And should it be?
	*/
	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult){
		if (add.rows != rowsOneCol.rows || add.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		final Matrix out = new Matrix(add.rows, add.cols);
		int offset = 0;
		for(int col=0; col<colMult; col++){
			for (int i = 0; i < rowsOneCol.w.length; i++){
				out.w[offset] = add.w[offset] + rowsOneCol.w[i];
				offset++;
			}
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					int offset = 0;
					for(int col=0; col<colMult; col++){
						for (int i = 0; i < rowsOneCol.w.length; i++) {
							
							//m1.dw[i] += out.dw[i];
							add.dw[offset] += out.dw[offset];
							
							//m2.dw[i] += out.dw[i];
							//FIXME this adjusts the bias colMult times more.
							//Should it be that vs just 1 times as much?
							//If did this with concatVectors of bias, which would it do?
							//Could be a problem if the dws arent changed equally?
							rowsOneCol.dw[i] += out.dw[offset];
							
							offset++;
						}
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	public Matrix oneMinus(final Matrix m) throws Exception {
		Matrix ones = Matrix.ones(m.rows, m.cols);
		Matrix out = sub(ones, m);
		return out;
	}
	
	public Matrix sub(final Matrix m1, final Matrix m2) throws Exception {
		Matrix out = add(m1, neg(m2));
		return out;
	}
	
	public Matrix smul(final Matrix m, final double s) throws Exception {
		Matrix m2 = Matrix.uniform(m.rows, m.cols, s);
		Matrix out = elmul(m, m2);
		return out;
	}
	
	public Matrix smul(final double s, final Matrix m) throws Exception {
		Matrix out = smul(m, s);
		return out;
	}
	
	public Matrix neg(final Matrix m) throws Exception {
		Matrix negones = Matrix.negones(m.rows, m.cols);
		Matrix out = elmul(negones, m);
		return out;
	}
	
	public Matrix elmul(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
		final Matrix out = new Matrix(m1.rows, m1.cols);
		for (int i = 0; i < m1.w.length; i++) {
			out.w[i] = m1.w[i] * m2.w[i];
		}
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					for (int i = 0; i < m1.w.length; i++) {
						m1.dw[i] += m2.w[i] * out.dw[i];
						m2.dw[i] += m1.w[i] * out.dw[i];
					}
				}
			};
			backprop.add(bp);
		}
		return out;
	}
}
