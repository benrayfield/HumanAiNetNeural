package mutable.recurrentjava.matrix;

public class MatrixStat{
	
	public final double[] radiusPerRow, radiusPerCol;
	
	public MatrixStat(double[] w, int rows, int cols){
		if(rows*cols != w.length) throw new Error("sizes not match");
		radiusPerRow = new double[rows];
		radiusPerCol = new double[cols];
		int offset = 0;
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				double sq = w[offset]*w[offset];
				radiusPerRow[r] += sq;
				radiusPerCol[c] += sq;
				offset++;
			}
		}
		for(int c=0; c<cols; c++){
			radiusPerCol[c] = Math.sqrt(radiusPerCol[c]);
		}
		for(int r=0; r<rows; r++){
			radiusPerRow[r] = Math.sqrt(radiusPerRow[r]);
		}
	}

}
