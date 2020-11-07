package immutable.rnn.recurrentjava;

public class RjPredictStep{
	
	//TODO create LearnStep, Lstm, NodesState, and PredictStep using the order of dims I want
	//instead of how recurrentjava does it, and translate between those and Rj* classes of similar name,
	//after get those working. Then when upgrade to opencl, wont inherit recurrentjava's dim order.
	//parallelIndex should be a more outer dim than a vec which can be used by itself.
	
	public final RjNodesState context;
	
	//public final double[][] vecs;
	public final MatD vecs;
	
	/** vecs is either input or output */
	public final boolean isInput;
	
	public RjPredictStep(boolean isInput, RjNodesState context, MatD vecs){
		this.isInput = isInput;
		this.context = context;
		this.vecs = vecs;
	}
	
	public RjPredictStep setOutVecs(MatD vecs){
		return new RjPredictStep(false, context, vecs);
	}
	
	public RjPredictStep setInVecs(MatD vecs){
		return new RjPredictStep(true, context, vecs);
	}
	
	/** until I figure out the rows vs cols of recurrentjava, will need this */
	public int howManyParallelVecs(){
		if(isInput) return vecs.cols; //FIXME is this backward?
		return vecs.cols; //FIXME is this backward?
	}
	

}
