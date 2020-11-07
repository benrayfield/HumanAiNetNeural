package immutable.rnn.recurrentjava;

/** immutable node states, what LSTM is thinking, excluding weights
which change slowly and are longterm memory
(TODO unless they're FastWeights like in hinton's paper which I might try after
I get lstm working better, and if so then the FastWeights would go in this class
too, and slow weights stay in Lstm.java).
<br><br>
2019-5-4 Lstm.java QUOTE
	//FIXME If Ive got these dim orders wrong it will learn mostly randomly.
	return new NodesState(
		//I plan to swap these dims in a future version cuz parallelIndex should be outer dim
		//except like in RBM learnloop when zigzagIndex is even more outer than parallelIndex,
		//but parallelIndex is still not the innermost dim in any of my code. That was a float[][][][].
		//Recurrentjava does it as double[vecContents][parallelIndex] and parallelIndex is always 1
		//(TODO change to allow parallel, unless it already works if use such a bigger Matrix and I dont know).
		new MatD(sizeIn, parallelVecs),
		//FIXME which order is it for output? Same as input?
		new MatD(sizeOut, parallelVecs)
	);
UNQUOTE.
*/
public class RjNodesState{
	
	//TODO create LearnStep, Lstm, NodesState, and PredictStep using the order of dims I want
	//instead of how recurrentjava does it, and translate between those and Rj* classes of similar name,
	//after get those working. Then when upgrade to opencl, wont inherit recurrentjava's dim order.
	//parallelIndex should be a more outer dim than a vec which can be used by itself.
	
	/** sometimes hidden from lstm outputs (gradually more or less) by outputGate.
	Each lstm node has 2 scalars of state.
	*/
	public final MatD hiddenContext;
	
	/** lstm outputs */
	public final MatD cellContext;
	
	public RjNodesState(MatD hiddenContext, MatD cellContext){
		this.hiddenContext = hiddenContext;
		this.cellContext = cellContext;
	}

}
