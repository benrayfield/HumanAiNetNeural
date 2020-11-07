package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.recurrentjava.flop.unary.SigmoidUnit;
import immutable.recurrentjava.flop.unary.TanhUnit;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;

public class LstmLayer implements Model {
	
	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	
	//benrayfield added this
	public final int parallelSize;
	
	/** w(eight) i(nput) o(utput) c(ontext) f(orget) (e)x(ternal) h(idden) b(ias) --comment by benrayfield.
	benrayfield made these public.
	*/
	public Matrix Wix, Wih, bi;
	public Matrix Wfx, Wfh, bf;
	public Matrix Wox, Woh, bo;
	public Matrix Wcx, Wch, bc;
	
	//benrayfield made these 2 public
	public Matrix hiddenContext;
	public Matrix cellContext;
	
	Unaflop fInputGate = new SigmoidUnit();
	Unaflop fForgetGate = new SigmoidUnit();
	Unaflop fOutputGate = new SigmoidUnit();
	Unaflop fCellInput = new TanhUnit();
	Unaflop fCellOutput = new TanhUnit();
	
	//benrayfield added this constructor
	public LstmLayer(
		int parallelSize,
		Matrix Wix, Matrix Wih,
		Matrix Wfx, Matrix Wfh,
		Matrix Wox, Matrix Woh,
		Matrix Wcx, Matrix Wch,
		Matrix bi, Matrix bf, Matrix bo, Matrix bc
	){
		this.parallelSize = parallelSize;
		this.inputDimension = Wix.cols;
		this.outputDimension = Wix.rows;
		this.Wix = Wix;
		this.Wih = Wih;
		this.Wfx = Wfx;
		this.Wfh = Wfh;
		this.Wox = Wox;
		this.Woh = Woh;
		this.Wcx = Wcx;
		this.Wch = Wch;
		this.bi = bi;
		this.bf = bf;
		this.bo = bo;
		this.bc = bc;
	}
	
	//benrayfield added parallelSize param
	public LstmLayer(int parallelSize, int inputDimension, int outputDimension, float initParamsStdDev, Random rng){
		this.parallelSize = parallelSize;
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		Wix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wih = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bi = new Matrix(outputDimension);
		Wfx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wfh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		//set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
		bf = Matrix.ones(outputDimension, 1);
		Wox = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Woh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bo = new Matrix(outputDimension);
		Wcx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wch = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bc = new Matrix(outputDimension);
	}
	
	//benrayfield (maybe todo?) editing this func to make it parallel (does it need edit?)
	public Matrix forward(Matrix input, Graph g) {
		
		//input gate
		Matrix sum0 = g.mul(Wix, input);
		Matrix sum1 = g.mul(Wih, hiddenContext);
		
		/*
		benrayfield
		FIXME adding 200x5 (correct) to 200x1 bias (also correct) so
		the wrong part must be that they're added. multiply maybe? Or what?
		Maybe the problem is just the bias? In the worst case
		I could expand the bias to parallelSize times bigger even though it
		would be duplicated, or just not use bias. so far i havent used any bias, always 0.
		The input has the right size, has the 5. And the weights are going to multiply along the 200
		so will probably align. But maybe recurrentjava automaticly changes the bias
		since g.add puts in a backprop Runnable, same as g.mul and g.elmul etc.
		Check if thats happening, and if not, temporarily remove bias until get this working.
		
		It is learning bias, so cant remove that:
		[RjLstm
			wix[MatD 200x27 ave-0.0014485719521050615_dev0.48788931575090083 stepCacheAve0.7423699175798519_dev6.025125831159763]
			wih[MatD 200x200 ave-0.003125636993955261_dev0.4605971213954811 stepCacheAve4.65404458763069_dev27.393380898104144]
			wfx[MatD 200x27 ave-0.08188680327688407_dev0.4220012439536712 stepCacheAve0.47719103616176517_dev5.23490061914481]
			wfh[MatD 200x200 ave-0.009350908915877833_dev0.44189911255833386 stepCacheAve4.575573546217935_dev27.67323888500485]
			wox[MatD 200x27 ave-0.012143024447715875_dev0.5124555903502763 stepCacheAve2.0514819789047953_dev9.097674671192907]
			woh[MatD 200x200 ave-0.003546678916927777_dev0.46789957278887007 stepCacheAve12.77031223921532_dev41.7198154914636]
			wcx[MatD 200x27 ave0.001075891347215641_dev0.6443046273563381 stepCacheAve5.654398993553076_dev61.19992891546001]
			wch[MatD 200x200 ave0.001151554778488889_dev0.44477712315895906 stepCacheAve32.35784326981915_dev185.20793598161598]
			outFeedforwardW[MatD 27x200 ave-0.055036937960596284_dev0.8600310176725333 stepCacheAve0.005002471081722408_dev0.030606592323331218]
			bi[MatD 200x1 ave-0.24206230095192222_dev0.3795893123360464 stepCacheAve23.682061626929276_dev79.6676400778197]
			bf[MatD 200x1 ave-0.42192077715074366_dev0.4174042074738025 stepCacheAve25.61485441989714_dev75.02930537012149]
			bo[MatD 200x1 ave-0.28108560423689466_dev0.3858313602951993 stepCacheAve55.521112125772035_dev95.49582021657078]
			bc[MatD 200x1 ave0.0400389308821018_dev0.35614722226464757 stepCacheAve170.9705436134705_dev522.1309309367507]
			outFeedforwardB[MatD 27x1 ave-2.7892798204474634_dev1.4508662742348142 stepCacheAve0.03415621470075293_dev0.11284659786562884]
			nmhpnull
		]
		
		Matrix sum2_test = g.mul(Wfx, input);
		Matrix sum4_test = g.mul(Wox, input);
		Matrix sum3_test = g.mul(Wfh, hiddenContext);
		Matrix forgetGate_withoutBias_test = g.nonlin(fForgetGate, g.add(sum2_test, sum3_test));
		
		Matrix sum6_test = g.mul(Wcx, input);
		Matrix sum7_test = g.mul(Wch, hiddenContext);
		Matrix cellInput_withoutBias_test = g.nonlin(fCellInput, g.add(sum6_test, sum7_test));
		
		cellInput_withoutBias_test is a 200x5 (or was it 5x200?), so the weights are duplicated only as many times
		as inputs. I dont think they should have to be duplicated at all since all I want is to
		sum the derivatives, but now that I think about it, derivatives backward summing them at each step
		would blur it together too much. but it works in RBM only duplicating the node states
		but nothing about the weights.
		What am I missing here?
		Am I mistaking weights for nodes? Hidden weights arrays are 200x200,
		and theres 200 nodes. So as long as its not a 200x200x5 (where parallelSize is 5 in this test)
		such as a 200x1000, that thing I'm worried about isnt happening.
		Its supposed to have 200xParallelindex arrays.
		
		Im considering in opencl reusing CLMem objects for the mutable Matrix objects
		since the whole LSTM could fit in gpu memory and do all the steps in 1 call that way,
		and maybe the same for these temp calculations as CLMems
		but hopefully the temp calculations can be done in memory local to each get_global_id.
		But it appears recurrentjava is creating more Matrix objects the deeper the datasequence gets,
		which are the temp vars, and I dont want to put all that in gpu memory at once.
		Maybe I should stick to OpenclUtil's existing way of doing 1 opencl kernel at a time
		and copying the memory into and out of gpu each time.
		*/
		
		/*Matrix sum2_test = g.mul(Wfx, input);
		Matrix sum4_test = g.mul(Wox, input);
		Matrix sum3_test = g.mul(Wfh, hiddenContext);
		Matrix forgetGate_withoutBias_test = g.nonlin(fForgetGate, g.add(sum2_test, sum3_test));
		
		Matrix sum6_test = g.mul(Wcx, input);
		Matrix sum7_test = g.mul(Wch, hiddenContext);
		Matrix cellInput_withoutBias_test = g.nonlin(fCellInput, g.add(sum6_test, sum7_test));
		*/
		
		//FIXME I must either make the bias bigger or add a new op to Graph that acts as if its bigger.
		
		int parallelSize = sum0.cols;
		Matrix inputGate = g.nonlin(
			fInputGate,
			g.add_rowsCols_to_rowsColsWithColmult(
				g.add(sum0, sum1),
				bi,
				parallelSize
			)
		);
		
		//forget gate
		Matrix sum2 = g.mul(Wfx, input);
		Matrix sum3 = g.mul(Wfh, hiddenContext);
		Matrix forgetGate = g.nonlin(
			fForgetGate,
			g.add_rowsCols_to_rowsColsWithColmult(
				g.add(sum2, sum3),
				bf,
				parallelSize
			)
		);
		
		//output gate
		Matrix sum4 = g.mul(Wox, input);
		Matrix sum5 = g.mul(Woh, hiddenContext);
		Matrix outputGate = g.nonlin(
			fOutputGate,
			g.add_rowsCols_to_rowsColsWithColmult(
				g.add(sum4, sum5),
				bo,
				parallelSize
			)
		);

		//write operation on cells
		Matrix sum6 = g.mul(Wcx, input);
		Matrix sum7 = g.mul(Wch, hiddenContext);
		Matrix cellInput = g.nonlin(
			fCellInput,
			g.add_rowsCols_to_rowsColsWithColmult(
				g.add(sum6, sum7),
				bc,
				parallelSize
			)
		);
		
		//compute new cell activation
		Matrix retainCell = g.elmul(forgetGate, cellContext);
		Matrix writeCell = g.elmul(inputGate,  cellInput);
		Matrix cellAct = g.add(retainCell,  writeCell);
		
		//compute hidden state as gated, saturated cell activations
		Matrix output = g.elmul(outputGate, g.nonlin(fCellOutput, cellAct));
		
		//rollover activations for next iteration
		hiddenContext = output;
		cellContext = cellAct;
		
		return output;
	}

	/*
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		//input gate
		Matrix sum0 = g.mul(Wix, input);
		Matrix sum1 = g.mul(Wih, hiddenContext);
		Matrix inputGate = g.nonlin(fInputGate, g.add(g.add(sum0, sum1), bi));
		
		//forget gate
		Matrix sum2 = g.mul(Wfx, input);
		Matrix sum3 = g.mul(Wfh, hiddenContext);
		Matrix forgetGate = g.nonlin(fForgetGate, g.add(g.add(sum2, sum3), bf));
		
		//output gate
		Matrix sum4 = g.mul(Wox, input);
		Matrix sum5 = g.mul(Woh, hiddenContext);
		Matrix outputGate = g.nonlin(fOutputGate, g.add(g.add(sum4, sum5), bo));

		//write operation on cells
		Matrix sum6 = g.mul(Wcx, input);
		Matrix sum7 = g.mul(Wch, hiddenContext);
		Matrix cellInput = g.nonlin(fCellInput, g.add(g.add(sum6, sum7), bc));
		
		//compute new cell activation
		Matrix retainCell = g.elmul(forgetGate, cellContext);
		Matrix writeCell = g.elmul(inputGate,  cellInput);
		Matrix cellAct = g.add(retainCell,  writeCell);
		
		//compute hidden state as gated, saturated cell activations
		Matrix output = g.elmul(outputGate, g.nonlin(fCellOutput, cellAct));
		
		//rollover activations for next iteration
		hiddenContext = output;
		cellContext = cellAct;
		
		return output;
	}
	*/

	@Override
	public void resetState(){
		/*
		//WARNING: this will break benrayfield's code unless these are changed to create parallelSize times bigger arrays
		hiddenContext = new Matrix(outputDimension);
		cellContext = new Matrix(outputDimension);
		*/
		//FIXME benrayfield experimental:
		hiddenContext = new Matrix(outputDimension, parallelSize);
		cellContext = new Matrix(outputDimension, parallelSize);
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(Wix);
		result.add(Wih);
		result.add(bi);
		result.add(Wfx);
		result.add(Wfh);
		result.add(bf);
		result.add(Wox);
		result.add(Woh);
		result.add(bo);
		result.add(Wcx);
		result.add(Wch);
		result.add(bc);
		return result;
	}
}
