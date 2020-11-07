package immutable.rnn.recurrentjava;

/** neuralnet ins and outs. Training is on a set of sequences of these,
such as a set of various partially overlapping time ranges
(which share arrays where overlap since used as immutable).
*/
public class RjLearnStep{
	
	//TODO create LearnStep, Lstm, NodesState, and PredictStep using the order of dims I want
	//instead of how recurrentjava does it, and translate between those and Rj* classes of similar name,
	//after get those working. Then when upgrade to opencl, wont inherit recurrentjava's dim order.
	//parallelIndex should be a more outer dim than a vec which can be used by itself.
	
	public final MatD ins;
	
	public final MatD outs;
	
	public RjLearnStep(MatD ins, MatD outs){
		this.ins = ins;
		this.outs = outs;
	}
	
	/** until I figure out the rows vs cols of recurrentjava, will need this */
	public int howManyParallelVecs(){
		return ins.cols; //FIXME is this backward?
	}

}
