package immutable.rnn.opencl;

import immutable.util.IntRange;
import immutable.util.MathUtil;
import immutable.util.Text;
import mutable.compilers.opencl.OpenclUtil;

/** This is a prototype to get working before OpenclGru. */
public class CpuGru extends Gru{
	
	//FIXME do I want to map weights, node states, etc, into one big int space
	//that unrollbackprop etc can happen in?

	public CpuGru(float[][][] weightHidden, float[][][] weightIn, float[][] biasHidden, IntRange outs){
		super(weightHidden, weightIn, biasHidden, outs);
	}
	
	public static String n = Text.n;
	
	public float[][] forward(float[][] nodeStates, float[][] ins, boolean returnAllNodes){
		if(!returnAllNodes) throw new Error("TODO");
		float[][] outNodeStates = MathUtil.newArraySameSizeAs(nodeStates, 0f);
		int vecs = nodeStates.length;
		for(int vec=0; vec<vecs; vec++){
			for(int to=0; to<nodes; to++){
				
				float sumMixB = biasHidden[GRU_MIX][to];
				float sumResetB = biasHidden[GRU_RESET][to];
				float sumNewB = biasHidden[GRU_NEW][to];
				
				float sumMixH = 0, sumResetH = 0, sumNewH = 0/*, sumNewGatedH = 0*/;
				for(int from=0; from<nodes; from++){
					float mul = nodeStates[vec][from];
					sumMixH += weightIn[GRU_MIX][from][to]*mul;
					sumResetH += weightIn[GRU_RESET][from][to]*mul;
					sumNewH += weightIn[GRU_NEW][from][to]*mul;
				}
				
				float sumMixI = 0, sumResetI = 0, sumNewI = 0;
				for(int in=0; in<nodes; in++){
					float mul = ins[vec][in];
					sumMixI += weightIn[GRU_MIX][in][to]*mul;
					sumResetI += weightIn[GRU_RESET][in][to]*mul;
					sumNewI += weightIn[GRU_NEW][in][to]*mul;
				}
				
				//float theMix = (float) MathUtil.sigmoid(sumMix);
				//float theReset = (float) MathUtil.sigmoid(sumReset);
				//float theNew = (float) Math.tanh(sumNew);
						
				//these var names partially copied from recurrentjava GruLayer.forward
				float context = nodeStates[vec][to];
				
				float sum0 = sumMixI;
				float sum1 = sumMixH;
				float actMix = (float) MathUtil.sigmoid(sum0+sum1+sumMixB);
				
				float sum2 = sumResetI;
				float sum3 = sumResetH;
				float actReset = (float) MathUtil.sigmoid(sum2+sum3+sumResetB);
				
				float sum4 = sumNewI;
				//float gatedContext = actReset*context;
				//float sum5 = sumNewH FIXME this isnt sumNewH cuz it needs to be from gatedContext, not context;
				float sum5 = sumNewH*actReset; //cuz sumNewH is (HHnew matmul context)
				float actNewPlusGatedContext = (float) Math.tanh(sum4+sum5+sumNewB);
				
				/*float sum4 = IHnew matmul input;
				float gatedContext = theReset*context;
				float sum5 = HHnew matmul gatedContext;
				//float actNewPlusGatedContext = (float)Math.tanh(sum4+sum5+Bnew);
				//float actNewPlusGatedContext = (float)Math.tanh(sum4+sum5+Bnew);
				float actNewPlusGatedContext = tanh((IHnew matmul input)+(HHnew matmul (reset*context))+Bnew)
				*/
						
				float output = context*actMix+(1-actMix)*actNewPlusGatedContext;
				
				//float memvals = theMix*context;
				//float newvals = (1-theMix)*actNewPlusGatedContext;
				//float output = memvals+newvals;
				
				outNodeStates[vec][to] = output; //next context
			}
		}
		return outNodeStates;
	}
	
	public float[][][] forwardN(
			float[][] firstNodeStates, float[][][] ins, boolean returnAllNodes){
		throw new Error("TODO");
	}

	public Gru learn(float[][] startNodeStates, LearnStep[] steps){
		// TODO Auto-generated method stub
		return null;
	}

}
