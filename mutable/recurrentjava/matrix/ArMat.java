package mutable.recurrentjava.matrix;

import java.io.Serializable;
import java.util.Random;

/** ARray MATrix. This is modified RecurrentJava Matrix.
Matrix class is now superclass of this and CLMatrix (opencl).
*/
@Deprecated //merging ArMat and ClMat back into Matrix with Buffer and lazyCreate CLMem
public class ArMat /*extends Matrix implements Serializable*/{
	
	/*
	//benrayfield changed double to float and (TODO) many places that depend on it
	//and is planning to replace each with pair<FloatBuffer,CLMem> (probably in some wrapper together).
	public float[] w;
	public float[] dw;
	public float[] stepCache;
	
	/** see RjOptions.testDelayedUpdateOfWeights.
	FIXME remove this after get those test results, or at least leave the array null.
	Instead of adding the weight changes directly into w[], add them here,
	and later move from here to w[].
	*
	public double[] testDelayedUpdateOfWeights;
	*
	
	@Override
	public String toString() {
		String result = "";
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				result += String.format("%.4f",getW(r, c)) + "\t";
			}
			result += "\n";
		}
		return result;
	}
	
	public Matrix clone() {
		ArMat result = new ArMat(rows, cols);
		for (int i = 0; i < w.length; i++) {
			result.w[i] = w[i];
			result.dw[i] = dw[i];
			result.stepCache[i] = stepCache[i];
		}
		return result;
	}

	public void resetDw() {
		for (int i = 0; i < dw.length; i++) {
			dw[i] = 0;
		}
	}
	
	public void resetStepCache() {
		for (int i = 0; i < stepCache.length; i++) {
			stepCache[i] = 0;
		}
	}
	
	public static ArMat transpose(ArMat m) {
		ArMat result = new ArMat(m.cols, m.rows);
		for (int r = 0; r < m.rows; r++) {
			for (int c = 0; c < m.cols; c++) {
				result.setW(c, r, m.getW(r, c));
			}
		}
		return result;
	}
	
	public static ArMat rand(int rows, int cols, float initParamsStdDev, Random rng) {
		ArMat result = new ArMat(rows, cols);
		for (int i = 0; i < result.w.length; i++) {
			result.w[i] = (float)(rng.nextGaussian() * initParamsStdDev);
		}
		return result;
	}
	
	public static ArMat ident(int dim) {
		ArMat result = new ArMat(dim, dim);
		for (int i = 0; i < dim; i++) {
			result.setW(i, i, 1f);
		}
		return result;
	}
	
	public static ArMat uniform(int rows, int cols, float s) {
		ArMat result = new ArMat(rows, cols);
		for (int i = 0; i < result.w.length; i++) {
			result.w[i] = s;
		}
		return result;
	}
	
	public static ArMat ones(int rows, int cols) {
		return uniform(rows, cols, 1f);
	}
	
	public static Matrix negones(int rows, int cols) {
		return uniform(rows, cols, -1f);
	}
	
	public ArMat(int dim){
		super(dim,1);
		this.w = new float[rows * cols];
		this.dw = new float[rows * cols];
		this.stepCache = new float[rows * cols];
		//if(RjOptions.testDelayedUpdateOfWeights){
		//	this.testDelayedUpdateOfWeights = new double[rows*cols];
		//}
	}
	
	public ArMat(int rows, int cols) {
		super(rows,cols);
		this.w = new float[rows * cols];
		this.dw = new float[rows * cols];
		this.stepCache = new float[rows * cols];
		//if(RjOptions.testDelayedUpdateOfWeights){
		//	this.testDelayedUpdateOfWeights = new double[rows*cols];
		//}
	}
	
	public ArMat(float[] vector) {
		super(vector.length,1);
		this.w = vector;
		this.dw = new float[vector.length];
		this.stepCache = new float[vector.length];
		//if(RjOptions.testDelayedUpdateOfWeights){
		//	this.testDelayedUpdateOfWeights = new double[vector.length];
		//}
	}
	
	private float getW(int row, int col) {
		return w[index(row, col)];
	}
	
	private void setW(int row, int col, float val) {
		w[index(row, col)] = val;
	}
	
	public void normByMaxRadius(float maxRadiusPerRow, float maxRadiusPerCol){
		normByMaxRadius(new MatrixStat(w,rows,cols), maxRadiusPerRow, maxRadiusPerCol);
	}
	
	/** benrayfield is adding funcs to measure and norm, such as by maxradius andOr L1 andOr L2 norm,
	but since theres stepCache (is that a kind of rmsprop?) norming each weight change on bellcurve
	of recent changes to that weight, I'll start with just maxradius since its idempotent of that.
	*
	public void normByMaxRadius(MatrixStat stat, float maxRadiusPerRow, float maxRadiusPerCol){
		if(maxRadiusPerRow <= 0 || maxRadiusPerCol <= 0) throw new Error("must be positive");
		int offset = 0;
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
				w[offset] *= Math.min(multCuzOfRow, multCuzOfCol); //always multiply by at most 1
				offset++;
			}
		}
	}
	
	/*public static double[] radiusPerRow(double[] w, int rows, int cols){
		
	}
	
	public static double[] radiusPerCol(double[] w, int rows, int cols){
	*/

}
