package mutable.recurrentjava.matrix;

import java.nio.FloatBuffer;

public class MatrixStat{
	
	public final float[] radiusPerRow, radiusPerCol;
	
	//public MatrixStat(float[] w, int rows, int cols){
	public MatrixStat(FloatBuffer w, int rows, int cols){
		if(rows*cols != w.capacity()) throw new Error("sizes not match");
		radiusPerRow = new float[rows];
		radiusPerCol = new float[cols];
		int offset = 0;
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				float wo = w.get(offset);
				float sq = wo*wo;
				//float sq = w[offset]*w[offset];
				radiusPerRow[r] += sq;
				radiusPerCol[c] += sq;
				offset++;
			}
		}
		for(int c=0; c<cols; c++){
			radiusPerCol[c] = (float)Math.sqrt(radiusPerCol[c]);
		}
		for(int r=0; r<rows; r++){
			radiusPerRow[r] = (float)Math.sqrt(radiusPerRow[r]);
		}
	}

}
