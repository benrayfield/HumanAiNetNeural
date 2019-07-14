package mutable.recurrentjava.matrix;
import java.io.Serializable;
import java.util.Random;

import mutable.recurrentjava.RjOptions;


public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	public int rows;
	public int cols;
	public double[] w;
	public double[] dw;
	public double[] stepCache;
	
	/** see RjOptions.testDelayedUpdateOfWeights.
	FIXME remove this after get those test results, or at least leave the array null.
	Instead of adding the weight changes directly into w[], add them here,
	and later move from here to w[].
	*/
	public double[] testDelayedUpdateOfWeights;
	
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
		Matrix result = new Matrix(rows, cols);
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
	
	public static Matrix transpose(Matrix m) {
		Matrix result = new Matrix(m.cols, m.rows);
		for (int r = 0; r < m.rows; r++) {
			for (int c = 0; c < m.cols; c++) {
				result.setW(c, r, m.getW(r, c));
			}
		}
		return result;
	}
	
	public static Matrix rand(int rows, int cols, double initParamsStdDev, Random rng) {
		Matrix result = new Matrix(rows, cols);
		for (int i = 0; i < result.w.length; i++) {
			result.w[i] = rng.nextGaussian() * initParamsStdDev;
		}
		return result;
	}
	
	public static Matrix ident(int dim) {
		Matrix result = new Matrix(dim, dim);
		for (int i = 0; i < dim; i++) {
			result.setW(i, i, 1.0);
		}
		return result;
	}
	
	public static Matrix uniform(int rows, int cols, double s) {
		Matrix result = new Matrix(rows, cols);
		for (int i = 0; i < result.w.length; i++) {
			result.w[i] = s;
		}
		return result;
	}
	
	public static Matrix ones(int rows, int cols) {
		return uniform(rows, cols, 1.0);
	}
	
	public static Matrix negones(int rows, int cols) {
		return uniform(rows, cols, -1.0);
	}
	
	public Matrix(int dim) {
		this.rows = dim;
		this.cols = 1;
		this.w = new double[rows * cols];
		this.dw = new double[rows * cols];
		this.stepCache = new double[rows * cols];
		if(RjOptions.testDelayedUpdateOfWeights){
			this.testDelayedUpdateOfWeights = new double[rows*cols];
		}
	}
	
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.w = new double[rows * cols];
		this.dw = new double[rows * cols];
		this.stepCache = new double[rows * cols];
		if(RjOptions.testDelayedUpdateOfWeights){
			this.testDelayedUpdateOfWeights = new double[rows*cols];
		}
	}
	
	public Matrix(double[] vector) {
		this.rows = vector.length;
		this.cols = 1;
		this.w = vector;
		this.dw = new double[vector.length];
		this.stepCache = new double[vector.length];
		if(RjOptions.testDelayedUpdateOfWeights){
			this.testDelayedUpdateOfWeights = new double[vector.length];
		}
	}
	
	private int index(int row, int col) {
		int ix = cols * row + col;
		return ix;
	}
	
	private double getW(int row, int col) {
		return w[index(row, col)];
	}
	
	private void setW(int row, int col, double val) {
		w[index(row, col)] = val;
	}
	
	public void normByMaxRadius(double maxRadiusPerRow, double maxRadiusPerCol){
		normByMaxRadius(new MatrixStat(w,rows,cols), maxRadiusPerRow, maxRadiusPerCol);
	}
	
	/** benrayfield is adding funcs to measure and norm, such as by maxradius andOr L1 andOr L2 norm,
	but since theres stepCache (is that a kind of rmsprop?) norming each weight change on bellcurve
	of recent changes to that weight, I'll start with just maxradius since its idempotent of that.
	*/
	public void normByMaxRadius(MatrixStat stat, double maxRadiusPerRow, double maxRadiusPerCol){
		if(maxRadiusPerRow <= 0 || maxRadiusPerCol <= 0) throw new Error("must be positive");
		int offset = 0;
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				double multCuzOfRow = 1;
				if(maxRadiusPerRow < stat.radiusPerRow[r]){
					multCuzOfRow = maxRadiusPerRow/stat.radiusPerRow[r];
				}
				double multCuzOfCol = 1;
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
