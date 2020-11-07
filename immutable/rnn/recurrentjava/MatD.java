package immutable.rnn.recurrentjava;

import java.util.Random;
import java.util.function.DoubleSupplier;

import immutable.util.DecayBell;
import immutable.util.MathUtil;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;

/** Imutable alternative to mutable.recurrentjava.matrix.Matrix.
<br><br>
Dont modify the array after first filling it. No other code than
what first fills an array is allowed to see a Mat cuz that would be mutable.
<br><br>
Index is cols*row + col. In 2d loops, do offset += cols instead of multiply.
<br><br>
For example, my Lstm.java uses this instead of recurrentjava's datastruct
and copies to recurrentjava, does calculations, then copies to a forkEdited result.
Im going to try upgrading recurrentJava to call opencl
instead of a bigger redesign where Lstm.java would call opencl.
Either way it will be done through OpenclUtil
and will do multiple vecs (such as 100) in parallel in the same
"DataSequence" (as recurrentjava calls it,
but I have my own immutable classes for those datastructs too.
*/
public class MatD{
	
	public final int rows, cols;
	
	public final float[] a;
	
	/** like in RecurrentJava Matrix stepCache.
	TODO find a different place to store this or way to do this,
	or at least store in twice as many MatD so only 1 array in each.
	Is this an adaptive learning rate per weight? Or is it neuralMomentum? What does it do?
	Is it like my DecayBell code?
	Its used in mutable.recurrentjava.trainer.Trainer:
	public static void updateModelParams(Model model, double stepSize) throws Exception {
		for (Matrix m : model.getParameters()) {
			for (int i = 0; i < m.w.length; i++) {
				
				// rmsprop adaptive learning rate
				double mdwi = m.dw[i];
				m.stepCache[i] = m.stepCache[i] * decayRate + (1 - decayRate) * mdwi * mdwi;
				
				// gradient clip
				if (mdwi > gradientClipValue) {
					mdwi = gradientClipValue;
				}
				if (mdwi < -gradientClipValue) {
					mdwi = -gradientClipValue;
				}
				
				// update (and regularize)
				m.w[i] += - stepSize * mdwi / Math.sqrt(m.stepCache[i] + smoothEpsilon) - regularization * m.w[i];
				m.dw[i] = 0;
			}
		}
	}
	public DecayBell add(double value, double decayFraction){
		if(decayFraction < 0 || 1 < decayFraction) throw new RuntimeException("decayFraction="+decayFraction);
		double newAve, newDev, newHowMuchData;
		if(howMuchData == 1){
			//This is caclulated as an infinite number of data points,
			//but instead of using a whole number for the quantity of data points,
			//the total number of data points is 1 and each is epsilon width.
			newAve = ave*(1-decayFraction) + decayFraction*value;
			double newDiff = value-ave;
			double sumOfSquares = dev*dev*(1-decayFraction) + decayFraction*newDiff*newDiff;
			//Don't divide sumOfSquares by quantity of data points because its 1.0
			newDev = Math.sqrt(sumOfSquares);
			newHowMuchData = 1;
		}else if(howMuchData == 0){
			newAve = value;
			newDev = 0;
			newHowMuchData = decayFraction;
		}else{
			newHowMuchData = howMuchData+decayFraction;
			newAve = (ave*howMuchData + decayFraction*value)/newHowMuchData;
			double newDiff = value-ave;
			double sumOfSquares = (dev*dev*howMuchData + decayFraction*newDiff*newDiff)/newHowMuchData;
			newDev = Math.sqrt(sumOfSquares);
			newHowMuchData = Math.min(1,newHowMuchData);
		}
		return new DecayBell(newAve, newDev, newHowMuchData);
	}
	*/
	public final float[] stepCache;
	
	public MatD(int rows, int cols){
		this(rows, new float[rows*cols], new float[rows*cols]);
	}
	
	public MatD(int rows, float[] a){
		this(rows, a, new float[a.length]);
	}
	
	public MatD(int rows, float[] a, float[] stepCache){
		this.rows = rows;
		this.cols = a.length/rows;
		this.a = a;
		this.stepCache = stepCache;
		if(a.length != rows*cols) throw new Error("not divisible");
		if(a.length != stepCache.length) throw new Error("Diff a[] and stepCache[] sizes");
	}
	
	public MatD(int rows, int cols, DoubleSupplier fill){
		this.rows = rows;
		this.cols = cols;
		this.a = new float[rows*cols];
		for(int i=0; i<a.length; i++) a[i] = (float)fill.getAsDouble();
		this.stepCache = new float[a.length];
	}
	
	/*Which of in, node, and out should have stepCache?
	This recurrentjava code it appear (TODO verify) used to create inputs uses an empty stepcache
	public Matrix(double[] vector) {
		this.rows = vector.length;
		this.cols = 1;
		this.w = vector;
		this.dw = new double[vector.length];
		this.stepCache = new double[vector.length];
	}*/
	
	/** use this until I figure out rows vs cols of how RecurrentJava does things,
	which I probably want to change cuz I think
	parallelIndex should always be a more outer dim than a single vec that could go in a LSTM by itself.
	FIXME: I'm unsure if rjInVec, rjOutVec, or rjNodeVec should have a nonzero stepcache.
	*/
	public static MatD rjInVec(float[] inVec){
		return new MatD(inVec.length, inVec, new float[inVec.length]); //FIXME is this backward?
	}
	
	/** see comment of rjInVec.
	FIXME: I'm unsure if rjInVec, rjOutVec, or rjNodeVec should have a nonzero stepcache.
	*/
	public static MatD rjOutVec(float[] outVec){
		return new MatD(outVec.length, outVec, new float[outVec.length]); //FIXME is this backward?
	}
	
	/** see comment of rjInVec. Example: bias of forget gate.
	FIXME: I'm unsure if rjInVec, rjOutVec, or rjNodeVec should have a nonzero stepcache.
	*/
	public static MatD rjNodeVec(float[] nodeVec){
		return new MatD(nodeVec.length, nodeVec, new float[nodeVec.length]); //FIXME is this backward?
	}
	
	public String toString(){
		float aAve = MathUtil.ave(a);
		float aDev = MathUtil.devGivenAve(aAve, a);
		float sAve = MathUtil.ave(stepCache);
		float sDev = MathUtil.devGivenAve(sAve, stepCache);
		return "[MatD "+rows+"x"+cols+" ave"+aAve+"_dev"+aDev+" stepCacheAve"+sAve+"_dev"+sDev+"]";
	}

}
