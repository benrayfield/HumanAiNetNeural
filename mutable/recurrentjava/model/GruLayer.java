package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.util.Text;
import immutable.recurrentjava.flop.unary.SigmoidUnit;
import immutable.recurrentjava.flop.unary.TanhUnit;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.autodiff.OpenclGraph;

/*
 * As described in:
 * 	"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
 * 	http://arxiv.org/abs/1406.1078
*/

public class GruLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	
	Matrix IHmix, HHmix, Bmix;
	Matrix IHnew, HHnew, Bnew;
	Matrix IHreset, HHreset, Breset;
	
	//benrayfield made this public
	public Matrix context;
	
	Unaflop fMix = new SigmoidUnit();
	Unaflop fReset = new SigmoidUnit();
	Unaflop fNew = new TanhUnit();
	
	/** context.cols */
	public final int parallelSize;
	
	public GruLayer(int parallelSize, int inputDimension, int outputDimension, float initParamsStdDev, Random rng) {
		this.parallelSize = parallelSize;
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		IHmix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHmix = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bmix = new Matrix(outputDimension);
		IHnew = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHnew = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bnew = new Matrix(outputDimension);
		IHreset = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHreset = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Breset= new Matrix(outputDimension);
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g){
		
		//FIXME call OpenclGraph.setIsTemp in add, mul, etc funcs???
		
		//FIXME this is always 0
		//System.out.println("context.w[0] = "+context.w[0]);
		//System.out.println("forward.START gru.context.w: = "+Text.stat(context.w.bufToFloatArray()));
		
		int par = input.cols;
		
		Matrix sum0 = g.mul(IHmix, input);
		Matrix sum1 = g.mul(HHmix, context);
		Matrix sum0_plus_sum1 = g.add(sum0, sum1);
		//Matrix sum0_plus_sum1 = g.add_rowsCols_to_rowsColsWithColmult(sum0, sum1, par);
		Matrix sum0_plus_sum1__plus_Bmix = g.add_rowsCols_to_rowsColsWithColmult(sum0_plus_sum1, Bmix, par);
		Matrix actMix = g.nonlin(fMix, sum0_plus_sum1__plus_Bmix);

		Matrix sum2 = g.mul(IHreset, input);
		Matrix sum3 = g.mul(HHreset, context);
		Matrix sum2_plus_sum3 = g.add(sum2, sum3);
		//Matrix sum2_plus_sum3 = g.add_rowsCols_to_rowsColsWithColmult(sum2, sum3, par);
		Matrix sum2_plus_sum3__plus_Breset = g.add_rowsCols_to_rowsColsWithColmult(sum2_plus_sum3, Breset, par);
		Matrix actReset = g.nonlin(fReset, sum2_plus_sum3__plus_Breset);
		
		Matrix sum4 = g.mul(IHnew, input);
		Matrix gatedContext = g.elmul(actReset, context);
		//Matrix gatedContext = g.elmult_rowsCols_to_rowsColsWithColmult(actReset, context, par);
		Matrix sum5 = g.mul(HHnew, gatedContext);
		Matrix sum4_plus_sum5 = g.add(sum4, sum5);
		Matrix sum4_plus_sum5__plus_Bnew = g.add_rowsCols_to_rowsColsWithColmult(sum4_plus_sum5, Bnew, par);
		Matrix actNewPlusGatedContext = g.nonlin(fNew, sum4_plus_sum5__plus_Bnew);
		
		Matrix memvals = g.elmul(actMix, context);
		//Matrix memvals = g.elmult_rowsCols_to_rowsColsWithColmult(actMix, context, par);
		Matrix newvals = g.elmul(g.oneMinus(actMix), actNewPlusGatedContext);
		Matrix output = g.add(memvals, newvals);
		
		//rollover activations for next iteration
		context = output; //context has parallelSize cols
		
		//System.out.println("forward.END gru.context.w: = "+Text.stat(context.w.bufToFloatArray()));
		
		return output;
	}
	
	/*
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		int par = input.cols;
		
		Matrix sum0 = g.mul(IHmix, input);
		Matrix sum1 = g.mul(HHmix, context);
		//Matrix sum0_plus_sum1 = g.add(sum0, sum1);
		Matrix sum0_plus_sum1 = g.add_rowsCols_to_rowsColsWithColmult(sum0, sum1, par);
		Matrix sum0_plus_sum1__plus_Bmix = g.add_rowsCols_to_rowsColsWithColmult(sum0_plus_sum1, Bmix, par);
		Matrix actMix = g.nonlin(fMix, sum0_plus_sum1__plus_Bmix);

		Matrix sum2 = g.mul(IHreset, input);
		Matrix sum3 = g.mul(HHreset, context);
		Matrix sum2_plus_sum3 = g.add_rowsCols_to_rowsColsWithColmult(sum2, sum3, par);
		Matrix sum2_plus_sum3__plus_Breset = g.add_rowsCols_to_rowsColsWithColmult(sum2_plus_sum3, Breset, par);
		Matrix actReset = g.nonlin(fReset, sum2_plus_sum3__plus_Breset);
		
		Matrix sum4 = g.mul(IHnew, input);
		//Matrix gatedContext = g.elmul(actReset, context);
		Matrix gatedContext = g.elmult_rowsCols_to_rowsColsWithColmult(actReset, context, par);
		Matrix sum5 = g.mul(HHnew, gatedContext);
		Matrix sum4_plus_sum5 = g.add(sum4, sum5);
		Matrix sum4_plus_sum5__plus_Bnew = g.add_rowsCols_to_rowsColsWithColmult(sum4_plus_sum5, Bnew, par);
		Matrix actNewPlusGatedContext = g.nonlin(fNew, sum4_plus_sum5__plus_Bnew);
		
		//Matrix memvals = g.elmul(actMix, context);
		Matrix memvals = g.elmult_rowsCols_to_rowsColsWithColmult(actMix, context, par);
		Matrix newvals = g.elmul(g.oneMinus(actMix), actNewPlusGatedContext);
		Matrix output = g.add(memvals, newvals);
		
		//rollover activations for next iteration
		context = output;
		CONTEXT now has par cols. I'm commentingout this whole func cuz I should have put bigger context.
		
		return output;
	}
	*/
	
	/*@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		Matrix sum0 = g.mul(IHmix, input);
		Matrix sum1 = g.mul(HHmix, context);
		Matrix sum0_plus_sum1 = g.add(sum0, sum1);
		Matrix actMix = g.nonlin(fMix, g.add(sum0_plus_sum1, Bmix));

		Matrix sum2 = g.mul(IHreset, input);
		Matrix sum3 = g.mul(HHreset, context);
		Matrix actReset = g.nonlin(fReset, g.add(g.add(sum2, sum3), Breset));
		
		Matrix sum4 = g.mul(IHnew, input);
		Matrix gatedContext = g.elmul(actReset, context);
		Matrix sum5 = g.mul(HHnew, gatedContext);
		Matrix actNewPlusGatedContext = g.nonlin(fNew, g.add(g.add(sum4, sum5), Bnew));
		
		Matrix memvals = g.elmul(actMix, context);
		Matrix newvals = g.elmul(g.oneMinus(actMix), actNewPlusGatedContext);
		Matrix output = g.add(memvals, newvals);
		
		//rollover activations for next iteration
		context = output;
		
		return output;
	}*/

	@Override
	public void resetState() {
		context = new Matrix(outputDimension, parallelSize);
		//context = new Matrix(outputDimension);
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(IHmix);
		result.add(HHmix);
		result.add(Bmix);
		result.add(IHnew);
		result.add(HHnew);
		result.add(Bnew);
		result.add(IHreset);
		result.add(HHreset);
		result.add(Breset);
		return result;
	}

}
