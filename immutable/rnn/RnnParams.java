package immutable.rnn;

/** Immutable. Recurrent neural network (rnn) params,
similar to RBM.java containing rbm params except
this doesnt also contain the neuralnet.
This will also work for GRU and LSTM which are kinds of RNN,
or a plain RNN which is just 1 weightedSum input and 1 sigmoid output.
*/
public class RnnParams{
	
	public final float learnRate;
	public final float rjTrainerDecayRate;
	public final float rjTrainerSmoothEpsilon;
	public final float rjTrainerGradientClipValue;
	public final float rjTrainerRegularization;
	
	/** starts with default values */
	public RnnParams(){
		this(
			.001f, //learnRate
			.999f, //rjTrainerDecayRate
			1e-8f, //rjTrainerSmoothEpsilon
			5f, //rjTrainerGradientClipValue
			.000001f //rjTrainerRegularization
		);
	}
	
	/** WARNING: The constructor will change often until I find the
	set of all RNN params I need, so you should instead
	construct a default RnnParams and set params 1 at a time,
	which is slower since it copies the other params
	with each param change but your code wont break when
	another param is added to the class.
	*/
	protected RnnParams(
		float learnRate,
		float rjTrainerDecayRate,
		float rjTrainerSmoothEpsilon,
		float rjTrainerGradientClipValue,
		float rjTrainerRegularization
	){
		this.learnRate = learnRate;
		this.rjTrainerDecayRate = rjTrainerDecayRate;
		this.rjTrainerSmoothEpsilon = rjTrainerSmoothEpsilon;
		this.rjTrainerGradientClipValue = rjTrainerGradientClipValue;
		this.rjTrainerRegularization = rjTrainerRegularization;
	}
	
	public RnnParams learnRate(float learnRate){
		return new RnnParams(learnRate, rjTrainerDecayRate, rjTrainerSmoothEpsilon, rjTrainerGradientClipValue, rjTrainerRegularization);
	}
	
	public RnnParams rjTrainerDecayRate(float rjTrainerDecayRate){
		return new RnnParams(learnRate, rjTrainerDecayRate, rjTrainerSmoothEpsilon, rjTrainerGradientClipValue, rjTrainerRegularization);
	}
	
	public RnnParams rjTrainerSmoothEpsilon(float rjTrainerSmoothEpsilon){
		return new RnnParams(learnRate, rjTrainerDecayRate, rjTrainerSmoothEpsilon, rjTrainerGradientClipValue, rjTrainerRegularization);
	}
	
	public RnnParams rjTrainerGradientClipValue(float rjTrainerGradientClipValue){
		return new RnnParams(learnRate, rjTrainerDecayRate, rjTrainerSmoothEpsilon, rjTrainerGradientClipValue, rjTrainerRegularization);
	}
	
	public RnnParams rjTrainerRegularization(float rjTrainerRegularization){
		return new RnnParams(learnRate, rjTrainerDecayRate, rjTrainerSmoothEpsilon, rjTrainerGradientClipValue, rjTrainerRegularization);
	}

}
