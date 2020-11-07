package mutable.learnloop_partsMovedToMutableuitoollearnloop.old;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import javax.swing.JPanel;

import immutable.rbm.learnloop.RBM;
import immutable.rbm.learnloop.Slide;
import mutable.util.Rand;
import mutable.util.Time;
import mutable.util.Var;

/** Wraps parts of PaintSlidingVecUi with more flexible params.
I'm planning to call this from mutable.uitools.learnloop.*
*/
public class RbmUitool extends JPanel{
	
	/*"TODO use Slide, SlideAction, LearningVec, etc (all immutable)."
	
	TODO should Slide have a Supplier<float[]> for next data?
	It could wrap a Function<Long,float[]> a static dataset generator
	or wrap interactive mouse input,
	but Slide etc would be mutable if the Supplier<float[]> was mutable.
	Cuz of that, I dont want them connected, just leave it as separate
	Supplier/Function and Slide.
	*/
	
	/*Do I want to bring in Slide and SlideAction etc?
	Yes, probably. I'll do occamsfuncer eventually but until then I feel the need for as much of it as possible,
	excluding maybe the "Function<Long,float[]> datasetGenerator", to be immutable.
	YES, AND...
	...
	...
	...
	I WANT IT DONE USING 2 NEW JAVACLASSES that are forkEditable immutable NavigableMap and List (avl or avllike),
	so dont have to deal with inefficiency of modifying List and map 1 at a time etc.
	Dont save on harddrive yet, except still some parts as json, so make it compatible with OccamsJsonDS.
	*/
	
	/** TODO use Var<RBM> and maybe Slide and SlideAction etc *
	protected RBM rbm
	*/
	
	/** either wraps a static dataset (such as Function<Long,float[]>)
	or an interactive data generator such as mouse movements.
	*/
	public final Supplier<float[]> datasetGenerator;
	
	/** the RBM and learning process, which learns what the Supplier<float[]> returns */
	public final Var<Slide> stateVar;
	
	public RbmUitool(Supplier<float[]> datasetGenerator, Var<Slide> stateVar){
	//public RbmUitool(Function<Long,float[]> datasetGenerator){
		this.datasetGenerator = datasetGenerator;
		this.stateVar = stateVar;
		
		/*TODO since Var<Slide> already contains RBM etc, theres very little to do here,
		but where should that be created? Maybe in the learnloop script a ==perBatch or ==perRbm
		section would tell the relevant vars to create it?
		*/
		
		//float[] prototypeData = datasetGenerator.apply(0L);
		float[] prototypeData = datasetGenerator.get();
		int visibleNodes = prototypeData.length;
		Random rand = Rand.strongRand;
		final float[][][] rbmEdges = RBM.newRandomEdges(rand, 0f, 50f, visibleNodes, visibleNodes, 400, 300);
		String learnFunc = //"===BEFORE=BATCH===\r\n"+
			//"String dataset = \"mnistOcrTestFile16x16From28x28ShrunkToHalfSizeAndOnehotLabelsAddedAlongASizeExpandingTo16x16\"\r\n"+
			//"int batchSize = 100;\r\n"+
			//"\r\n"+
			//"===PER=WEIGHT===\r\n"+
			"float att = tolowNodeAtt*tohighNodeAtt;\r\n" + 
			"float diff = learnRate*att*(toLearn-toUnlearn);\r\n" + 
			"float diffScaled = diff/batchSize;\r\n"+ //spread learnRates across vecs instead of each.\r\n" + 
			"float decay = weightDecay*diffScaled*diffScaled;\r\n" + 
			"float deriv = diffScaled - decay*theWeight;\r\n" + 
			"float dt = 1;\r\n"+ //FIXME this was the old code but its buggy cuz when learnRate changes, weightVelocityDecay does not, but learnRate*weightVelocityDecay would.\r\n"+ 
			"returnWevel = theWevel*(1-dt*wevelDecay) + dt*deriv;\r\n" + 
			"returnWeight = theWeight + dt*theWevel;";
		//int lowNNolaysAreScalar = 0;
		int nolays = rbmEdges.length+1;
		boolean[] nolayIsScalar = new boolean[nolays];
		float[][][] nmhp = new float[nolays][][];
		for(int i=0; i<nolays; i++){
			int nolaySize = RBM.nolaySize(rbmEdges, i);
			nmhp[i] = new float[nolaySize][nolaySize];
		}
		float[][] biasPerNodeside = new float[nolays*2][];
		float[][] attLev2PerNodeside = new float[nolays*2][];
		float bias = -.5f;
		for(int i=0; i<nolays*2; i++){
			int nolay = i/2; //down and up directions, or could set those separately
			int nolaySize = RBM.nolaySize(rbmEdges, nolay);
			attLev2PerNodeside[i] = new float[nolaySize];
			Arrays.fill(attLev2PerNodeside[i], 1f/attLev2PerNodeside[i].length); //make attLev2 sum to 1 per nolay. TODO later explore other variations of it
			biasPerNodeside[i] = new float[nolaySize];
			Arrays.fill(biasPerNodeside[i], bias);
		}
		int zigzagPredict = 10; //TOOD
		int zigzagLearn = zigzagPredict+1;
		int zigzagNorm = zigzagLearn;
		float[][] targetNodeAve = RBM.newZerosArraysSameSizeAsNodes(rbmEdges);
		for(float[] a : targetNodeAve) Arrays.fill(a, .25f);
		float[][][] zigzagLearnArray = new float[zigzagLearn][][];
		for(int z=0; z<zigzagLearnArray.length; z++) zigzagLearnArray[z] = RBM.newZerosArraysSameSizeAsNodes(rbmEdges);
		RBM rbm = new RBM()
			.setComment("RBM"+Time.now())
			.setNmhp(nmhp)
			.setBiasPerNodeside(biasPerNodeside)
			.setAttLev2PerNodeside(attLev2PerNodeside)
			.setLearnFunc(learnFunc)
			.setNolayIsScalar(nolayIsScalar)
			.setLearnByIncompleteBoltzenCode(false)
			.setAttLev1RelRangeForLearn(.5f)				
			.setAttLev1RelRangeForPredict(.5f)
			.setTargetNodeAve(targetNodeAve) //this code does nothing (hasnt for maybe a year as of Y2018M4)
			.setWeight(rbmEdges)
			.setWeightVelocity(RBM.newEmptyArraySameSizesAs(rbmEdges))
			.setWeightDecay(.02f)
			.setWeightVelocityDecay(.2f)
			.setZigzagsLearn(zigzagLearn)
			.setLearnRate(1e5f);
	}
	

}
