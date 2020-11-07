package mutable.mathExperiments;
import static mutable.util.Lg.*;
import java.awt.Graphics;
import java.util.function.IntBinaryOperator;

import javax.swing.JPanel;

import mutable.util.ColorUtil;
import mutable.util.ui.ScreenUtil;
import mutable.util.ui.StretchVideo;

/** the number of paths from a cell, zigzagging downleft vs downright,
to another certain cell is (t choose x), where t is number of rows down,
and x is number of cells from leftmost cell in bottom row.
Paint that.
*/
public class PascalsTriangleOnlyPathsFromAndToCertainCellTChooseX extends StretchVideo{
	
	protected int t, x;
	
	/** grid[0][0] is top of pascalstri. grid[2][0] and grid[1][1] and grid[0][2] are row2, so manhattanDistance.
	Only about half of the grid is used. The other half is there just cuz painting a rectangle is easier than a triangle.
	*/
	protected double[][] grid;
	
	//protected double[][] normedGrid;
	
	protected boolean doTChooseX;
	
	/** if not doTChooseX then is normal pascalstri */
	public PascalsTriangleOnlyPathsFromAndToCertainCellTChooseX(boolean doTChooseX, int t, int x){
		super(false, t+1, t+1, null);
		this.doTChooseX = doTChooseX;
		if(t < x) throw new Error(t+"==t < x=="+x);
		this.t = t;
		this.x = x;
		grid = new double[t+1][t+1];
		painter =  (IntBinaryOperator)(int Y, int X)->{
			//return ColorUtil.monochrome((float)normedGrid[Y][X]);
			return ColorUtil.monochrome((float)grid[Y][X]);
		};
		fill(doTChooseX, grid, 0, 0, t, x);
	}
	
	/** fixme can only call this once else will create wrong data */
	protected void fill(boolean doTChooseX, double[][] grid, int fromY, int fromX, int toY, int toX){
		//FIXME this is ignoring the from and to x and y's. I'm trying to start at (0,0) and go to a certain point first.
		if(doTChooseX){ //part of pascalstri that goes from and to 2 certain places
			int toPascalstriRow = toY-toX;
			int toPascalstriCol = toX;
			int dy = toY-fromY, dx = toX - fromX;
			grid[0][0] = 1;
			for(int pascalstriRow=0; pascalstriRow<t; pascalstriRow++){ //exclude last row which is t. write into row below.
				for(int pascalstriCol = 0; pascalstriCol<=pascalstriRow; pascalstriCol++){
					int x = pascalstriCol;
					int y = pascalstriRow-x;
					//double norm = 1;
					double norm = 1.02;
					//double here = choose(pascalstriRow,pascalstriCol);
					double here = grid[y][x];
					
					//THIS IS WRONG. its not displaying ONLY the paths that go to the chosen square at last pascalstriRow.
					//double chooseDownLeft = choose(pascalstriRow+1,pascalstriCol);
					//double chooseDownRight = choose(pascalstriRow,pascalstriCol+1);
					int downLeftY = y;
					int downLeftX = x+1;
					int downRightY = y+1;
					int downRightX = x;
					//double chooseDownLeft = choose(100-pascalstriRow, toX-downLeftX);
					//double chooseDownRight = choose(100-pascalstriRow, toX-downRightX);
					double chooseDownLeft = choose(100-pascalstriRow-1, 100-toPascalstriCol);
					double chooseDownRight = choose(100-pascalstriRow-1, 100-toPascalstriCol-1);
					
					double chooseSum = (chooseDownLeft+chooseDownRight);
					grid[y+1][x] += here*chooseDownLeft/chooseSum*norm;
					grid[y][x+1] += here*chooseDownRight/chooseSum*norm;
					//grid[y+1][x] += chooseDownLeft*.0000000000001;
					//grid[y][x+1] += chooseDownRight*.0000000000001;
				}
			}
		}else{ //normal pascalstri
			grid[0][0] = 1;
			for(int pascalstriRow=0; pascalstriRow<t; pascalstriRow++){ //exclude last row which is t. write into row below.
				for(int pascalstriCol = 0; pascalstriCol<=pascalstriRow; pascalstriCol++){
					int x = pascalstriCol;
					int y = pascalstriRow-x;
					double add = grid[y][x]/2;
					grid[y+1][x] += add;
					grid[y][x+1] += add;
				}
			}
		}
	}
	
	public static void main(String[] args) {
		lg("5 choose 2 = "+choose(5,2));
		lg("17 choose 11 = "+choose(17,11));
		ScreenUtil.testDisplayWithExitOnClose(new PascalsTriangleOnlyPathsFromAndToCertainCellTChooseX(true, 100, 80));
	}
	
	/** (y choose x) aka y!/((y-x)!*x!) */
	public static double choose(int y, int x){
		double ret = 1;
		for(int i=x+1; i<=y; i++) ret *= i;
		for(int i=1; i<=y-x; i++) ret /= i;
		return ret;
	}

}
