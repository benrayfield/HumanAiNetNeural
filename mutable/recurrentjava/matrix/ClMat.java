package mutable.recurrentjava.matrix;

import java.nio.FloatBuffer;

import mutable.compilers.opencl.BiMem;

/** openCL MATrix,
a GPU number crunching alternative to ArMat which uses float[].
There arent funcs here, like ArMat has funcs,
cuz those have to be done in opencl.
<br><br>
TODO Parts of recurrentjava will be ported to opencl, especially
those in Trainer, GruLayer, LstmLayer, FeedforwardLayer, and Matrix.
*/
@Deprecated //merging ArMat and ClMat back into Matrix with Buffer and lazyCreate CLMem
public class ClMat /*extends Matrix*/{
	
	/** the main data *
	public BiMem<FloatBuffer> w;
	
	/** backprop of data *
	public BiMem<FloatBuffer> dw;
	
	/** decaying sumOfSquares and L2 norming per weight
	(but in opencl I might use this for a variety of kinds of norming)
	used by mutable.recurrentjava.trainer.Trainer
	in Matrixs returned by
	List<Matrix> mutable.recurrentjava.model.Model.getParameters()
	but is not used in temporary Matrixs like those created in
	mutable.recurrentjava.model.GruLayer.forward(Matrix,Graph).
	*
	public BiMem<FloatBuffer> stepCache;
	
	public ClMat(int rows, int cols, boolean hasStepCache){
		super(rows,cols);
		int size = rows*cols;
		w = new BiMem(float.class, size);
		dw = new BiMem(float.class, size);
		if(hasStepCache){
			stepCache = new BiMem(float.class, size);
		}
	}*/

}
