package immutable.rbm.learnloop;
import mutable.util.MutDecayBell; //FIXME this cant be in immutable package
import immutable.rbm.learnloop.VecType;
import immutable.util.DecayBell;
import immutable.util.MathUtil;
import mutable.util.Rand; //FIXME this cant be in immutable package
import mutable.util.Time; //FIXME this cant be in immutable package

public class LearningVec{
	
	/** visibleNodes. Use as immutable */
	public final float[] vec;
	
	/** If true, is a random vec used to avoid learning identityFunc *
	public final boolean isRandom = false;
	
	/** false if is part of the test dataset, which is predicted to know accuracy on things that should
	be indirectly learned by inference from combos of things directly learned (invarep).
	FIXME this var is ignored as of 2018-4-12-1p except for display of it.
	*
	public final boolean enableLearningIfNonrandom = true;
	//public boolean enableLearning = MathUtil.weightedCoinFlip(.7);
	*
	public boolean shouldLearn(){
		return enableLearningIfNonrandom && !isRandom;
	}*/
	public final VecType vecType;
	
	/** Slidinglearnrandvecui displays in this order.
	Since this is an immutable/forkEditable class, timeCreated refers to float[] vec, not this object.
	*/
	public final double timeCreated;
	
	/** time predicted by a RBM */
	public final double timePredicted;
	
	public final DecayBell aveDiffOfErr;
	
	/** [zigzag][nolay][which vec but in this case is size 1][node]. This could be from cpu or gpu.
	TODO Modify RbmPanel to use this datastruct instead of the earlier design [zigzag][nolay][node].
	*/
	public final float[][][][] lastPredict;
	
	/** [zigzag][nolay][which vec but in this case is size 1][node]. This could be from cpu or gpu.
	TODO Modify RbmPanel to use this datastruct instead of the earlier design [zigzag][nolay][node].
	*/
	public final float[][][][] lastLearn;
	
	//public double stdDevOfErr = 1;
	//public final double aveDiffOfErr = 1;
	
	public LearningVec(){
		this(new float[0]);
	}
	
	public LearningVec(float[] vec){
		this(vec, VecType.vecForLearn);
	}
	
	public LearningVec(float[] vec, VecType vecType){
		this(vec, vecType, Time.now(), 0, new DecayBell());
	}
	
	public LearningVec(float[] vec, VecType vecType, double timeCreated, double timePredicted, DecayBell aveDiffOfErr){
		this.vec = vec;
		this.vecType = vecType;
		this.timeCreated = timeCreated;
		this.timePredicted = timePredicted;
		this.aveDiffOfErr = aveDiffOfErr;
		this.lastPredict = this.lastLearn = null;
	}
	
	public LearningVec(float[] vec, VecType vecType, double timeCreated, double timePredicted, DecayBell aveDiffOfErr, float[][][][] lastPredict, float[][][][] lastLearn){
		this.vec = vec;
		this.vecType = vecType;
		this.timeCreated = timeCreated;
		this.timePredicted = timePredicted;
		this.aveDiffOfErr = aveDiffOfErr;
		this.lastPredict = lastPredict;
		this.lastLearn = lastLearn;
	}
	
	public LearningVec setVec(float[] vec){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setVecType(VecType vecType){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setTimeCreated(double timeCreated){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setTimePredicted(double timePredicted){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setAveDiffOfErr(DecayBell aveDiffOfErr){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setLastPredict(float[][][][] lastPredict){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	public LearningVec setLastLearn(float[][][][] lastLearn){
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr, lastPredict, lastLearn);
	}
	
	/** does not update vec since thats constant */
	public LearningVec update(double timePredicted, float[] rbmOut){
		//this.timePredicted = timePredicted;
		//stdDevOfErr = RBM.stdDevOfDiff(vec, rbmOut);
		//aveDiffOfErr = RBM.aveDiff(vec, rbmOut);
		//aveDiffOfErrDecayBell.add(aveDiffOfErr, .2); //TODO make decay a param
		
		double incomingAveDiff = RBM.aveDiff(vec,rbmOut);
		double decay = .2; //TODO make decay a param
		return new LearningVec(vec, vecType, timeCreated, timePredicted, aveDiffOfErr.add(incomingAveDiff, decay));
	}
	
	/** does not update vec since thats constant */
	public LearningVec update(float[] rbmOut){
		return update(Time.now(), rbmOut);
	}

}
