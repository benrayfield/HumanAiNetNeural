package mutable.recurrentjava.matrix;
import java.io.Serializable;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Random;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CLMem;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.DependParam;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.Mem;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.SyMem;
import mutable.compilers.opencl.BiMem;
import mutable.compilers.opencl.FMem;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
//import mutable.compilers.opencl.connectors.lwjgl.Lwjgl;
import mutable.recurrentjava.RjOptions;

/** benrayfield changed the float[]s to FMem then generalized to MemInfo
so they could be either FMem (containing FloatBuffer to use in CPU)
or DependParam (containing no memory but to be used with pool of CLMems).
<br><br>
FSyMem (with a DependParam SYmbol) are used instead of some other kind of Mem without a DependParam.
The DependParams are how OpenclUtil.callOpenclDependnet refers to CLMem objects
that a FloatBuffer is not required for but some (inputs and outputs to opencl) have it.
*/
public class Matrix implements Cloneable{
	
	//FIXME convert everything back to NavigableMap instead of SortedMap, cuz SortedMap doesnt guarantee a backing SortedSet of keys.

	public final int rows, cols;
	
	/** rows*cols */
	public final int size;
	
	//public final boolean lazy;
	
	public final NavigableMap<String,FSyMem> mems;
	
	/** Example keys: "w", "dw", "stepCache", but generalizing it to any name for experiments in new neuralnet types */
	public FSyMem mem(String key){
		FSyMem ret = mems.get(key);
		if(ret == null){
			ret = newMem();
			mems.put(key, ret);
		}
		return ret;
	}
	
	/** same key as mem(String key). Creates if not exist. */
	public FloatBuffer buf(String key){
		return mem(key).mem();
	}
	
	/** symbol, same param as buf(String) and mem(String) */
	public DependParam sy(String key){
		return mem(key).sy;
	}
	
	/** unmodifiable but mutable, as mem(String key) can add to this set.
	All params of mem(String key) ever called here, which each created a FSyMem.
	The FSyMem is lazy of creating FloatBuffer as needed but nonlazy of creating DependParam.
	This is useful for naming CLMems andOr FloatBuffers by the same DependParam but not having
	to create duplicate buffers except for inputs and outputs but not most of them which are temp calculations.
	*
	public final SortedSet<String> keys = Collections.unmodifiableSortedSet(mems.navigableKeySet());
	*/
	
	
		
	/** the main data *
	public final FSyMem w;
	
	/** backprop of data *
	public final FSyMem dw;
	
	/** In recurrentjava, only Matrixs that are in a
	mutable.recurrentjava.model.Model.getParameters() use stepCache,
	so this is lazyCreate.
	<br><br>
	Decaying sumOfSquares and L2 norming per weight
	(but in opencl I might use this for a variety of kinds of norming)
	used by mutable.recurrentjava.trainer.Trainer
	in Matrixs returned by
	List<Matrix> mutable.recurrentjava.model.Model.getParameters()
	but is not used in temporary Matrixs like those created in
	mutable.recurrentjava.model.GruLayer.forward(Matrix,Graph).
	*
	public FSyMem stepCache(){
		if(stepCache == null){
			stepCache = newMem();
		}
		return stepCache;
	}
	public boolean hasStepCacheYet(){ return stepCache != null; }
	private FSyMem stepCache;
	*/
	
	public FSyMem newMem(){
		return new FSyMem("noComment"+Rand.strongRand.nextLong(), rows*cols);
		/*final int siz = rows*cols;
		return new FSyMem(
			new DependParam(float.class,siz),
			(int size)->BufferUtils.createFloatBuffer(siz)
		);*/
		//return lazy ? new DependParam(float.class, rows*cols) : new FMem(rows*cols);
	}
	
	public Matrix(float[] w){
		this(w.length);
		this.mem("w").put(w);
		//FloatBuffer buf = ((SyMem<FloatBuffer>)this.w).mem();
		//this.w.buf.position(0);
		//this.w.buf.put(w);
	}
	
	public Matrix(int rows, int cols){
	//public Matrix(boolean lazy, int rows, int cols){
		//this.lazy = lazy;
		this(rows,cols,new TreeMap());
		//w = newMem(); //FIXME use map instead
		//dw = newMem();
	}
	
	/** NavigableMap must be mutable */
	public Matrix(int rows, int cols, NavigableMap<String,FSyMem> mems){
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;
		this.mems = mems;
	}
	
	/*public Matrix(int rows, int cols, FSyMem w, FSyMem dw, FSyMem stepCache){
		//lazy = w.lazy();
		//if(dw.lazy() != lazy || (stepCache!=null && stepCache.lazy() != lazy))
		//	throw new Error("All must be lazy or all nonlazy");
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;
		//this.w = w; //FIXME use map instead
		//this.dw = dw;
		//this.stepCache = stepCache;
	}*/
	
	public Matrix(int dim){
		this(dim,1);
	}
	
	public int index(int row, int col){
		return cols*row + col;
	}
	
	public static Matrix uniform(int rows, int cols, float s){
		Matrix result = new Matrix(rows, cols);
		FSyMem resultW = result.mem("w");
		for (int i = 0; i < result.size; i++) {
			resultW.put(i, s);
		}
		return result;
	}
	
	public static Matrix ones(int rows, int cols) {
		return uniform(rows, cols, 1f);
	}
	
	public static Matrix negones(int rows, int cols) {
		return uniform(rows, cols, -1f);
	}
	
	public void normByMaxRadius(float maxRadiusPerRow, float maxRadiusPerCol){
		normByMaxRadius(new MatrixStat(buf("w"),rows,cols), maxRadiusPerRow, maxRadiusPerCol);
	}
	
	/** benrayfield is adding funcs to measure and norm, such as by maxradius andOr L1 andOr L2 norm,
	but since theres stepCache (is that a kind of rmsprop?) norming each weight change on bellcurve
	of recent changes to that weight, I'll start with just maxradius since its idempotent of that.
	*/
	public void normByMaxRadius(MatrixStat stat, float maxRadiusPerRow, float maxRadiusPerCol){
		if(maxRadiusPerRow <= 0 || maxRadiusPerCol <= 0) throw new Error("must be positive");
		int offset = 0;
		FSyMem w = mem("w");
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				float multCuzOfRow = 1;
				if(maxRadiusPerRow < stat.radiusPerRow[r]){
					multCuzOfRow = maxRadiusPerRow/stat.radiusPerRow[r];
				}
				float multCuzOfCol = 1;
				if(maxRadiusPerCol < stat.radiusPerRow[r]){
					multCuzOfCol = maxRadiusPerCol/stat.radiusPerCol[c];
				}
				w.putMult(offset, Math.min(multCuzOfRow, multCuzOfCol)); //always multiply by at most 1
				offset++;
			}
		}
	}
	
	public static Matrix rand(int rows, int cols, float initParamsStdDev, Random rng) {
		Matrix result = new Matrix(rows, cols);
		FSyMem resultW = result.mem("w");
		for (int i = 0; i < result.size; i++) {
			resultW.put(i, (float)(rng.nextGaussian() * initParamsStdDev));
		}
		return result;
	}
	
	public static Matrix ident(int dim) {
		Matrix result = new Matrix(dim, dim);
		FSyMem resultW = result.mem("w");
		for (int i = 0; i < dim; i++){
			//TODO optimize by creating a func similar to set(int,int,float) but which somehow doesnt call Matrix.mem("w") every time.
			resultW.put(result.index(i,i), 1f);
		}
		return result;
	}
	
	/*private void setW(int row, int col, float val) {
		FIXME this could get very inefficient cuz have to get from Map every time
		
		w.put(index(row, col), val);
	}*/
	
	/** Clone is deep cuz implemented it that way here, but in java is normally shallow */
	public Object clone(){
		NavigableMap<String,FSyMem> newMems = new TreeMap();
		for(Map.Entry<String,FSyMem> entry : mems.entrySet()){
			newMems.put(entry.getKey(), (FSyMem)entry.getValue().clone());
		}
		return new Matrix(rows, cols, newMems);
	}
	
}
