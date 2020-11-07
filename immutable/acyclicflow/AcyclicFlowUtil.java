package immutable.acyclicflow;

import java.util.Random;

public strictfp class AcyclicFlowUtil{
	
	public static final byte plus = 0, mult = 1, neg = 2, sine = 3, copy = 4, FUNCS = 5;
	
	public static int op(int firstPointer, int secondPointer, int funcIndex){
		if(firstPointer < 0 || firstPointer >= 4096) throw new Error("firstPointer="+firstPointer);
		if(secondPointer < 0 || secondPointer >= 4096) throw new Error("secondPointer="+secondPointer);
		if(funcIndex < 0 || funcIndex >= 256) throw new Error("funcIndex="+funcIndex);
		return (firstPointer<<20)|(secondPointer<<8)|funcIndex;
	}
	
	/** Reads int[] ops. Reads and writes double[] state.
	double[] state has 4 sections in this order: inputs, state0, temp, state1.
	Updates each value in state in order of indexs of state.
	Inputs should copy themself.
	state0 should copy state1.
	temp should derive its value only from lower index,
	including idnexs in temp, state0, and inputs.
	Inputs can optionally be replaced by external process before calling this,
	such as from microphone amplitude or mouse position.
	<br><br>
	Max array size is 4096. Each int has 2 pointers into that and 8 bit funcIndex.
	This is meant to fit in L1 cpu cache and allow low lag hotswapping of music tools.
	*
	public static void acyclicFlow(double[] state, int[] ops){
		if(state.length != ops.length) throw new Error("diff sizes");
		if(state.length > (1<<12)) throw new Error("too big: "+state.length);
		for(int i=0; i<state.length; i++){
			int op = ops[i];
			double firstScalar = state[op>>>20];
			double secondScalar = state[(op>>>8)&0xfff];
			double ret;
			switch((byte)op){
			case plus:
				ret = firstScalar+secondScalar;
			break;
			case mult:
				ret = firstScalar*secondScalar;
			break;
			case neg:
				ret = -firstScalar;
			break;
			case sine:
				ret = Math.sin(firstScalar);
			break;
			case copy:
				ret = firstScalar;
			break; 
			default:
				ret = 0;
			}
			state[i] = ret;
		}
	}*/
	public static void acyclicFlow(float[] state, int[] ops){
		if(state.length != ops.length) throw new Error("diff sizes");
		if(state.length > (1<<12)) throw new Error("too big: "+state.length);
		for(int i=0; i<state.length; i++){
			int op = ops[i];
			float firstScalar = state[op>>>20];
			float secondScalar = state[(op>>>8)&0xfff];
			float ret;
			switch((byte)op){
			case plus:
				ret = firstScalar+secondScalar;
			break;
			case mult:
				ret = firstScalar*secondScalar;
			break;
			case neg:
				ret = -firstScalar;
			break;
			case sine:
				ret = (float)Math.sin(firstScalar);
			break;
			case copy:
				ret = firstScalar;
			break; 
			default:
				ret = 0;
			}
			state[i] = ret;
		}
	}
	
	public static void main(String[] args){
		int[] ops = new int[1000];
		float[] state = new float[ops.length];
		int inputSize = 1;
		for(int i=0; i<inputSize; i++){
			ops[i] = op(i,0,copy);
		}
		int stateSize = 40;
		for(int i=0; i<stateSize; i++){
			ops[inputSize+i] = op(ops.length-stateSize+i,0,copy);
		}
		Random r = new Random();
		for(int i=inputSize; i<ops.length; i++){
			ops[i] = op(r.nextInt(i),r.nextInt(i),r.nextInt(FUNCS));
			//only makes it 20% faster if 500 cycles and 1000 ops per cycle: if((byte)ops[i] == sine) ops[i] = plus; //TESTING removing the slowest op
		}
		int cycles = 500;
		acyclicFlow(state, ops); //dont count classloading etc in benchmark
		for(int j=0; j<cycles; j++){ //dont count JIT optimization in benchmark
			acyclicFlow(state, ops);
			System.arraycopy(state, state.length-stateSize, state, inputSize, stateSize); //copy state from end to just after inputs
		}
		long timeStart = System.nanoTime();
		for(int j=0; j<cycles; j++){
			acyclicFlow(state, ops);
			System.arraycopy(state, state.length-stateSize, state, inputSize, stateSize); //copy state from end to just after inputs
		}
		long timeEnd = System.nanoTime();
		double duration = 1e-9*(timeEnd-timeStart);
		double opsPerSec = ops.length*cycles/duration;
		double cyclesPerSec = cycles/duration;
		System.out.println("opsPerSec="+opsPerSec+" and cyclesPerSec="+cyclesPerSec+" when cycles="+cycles+" and ops.length="+ops.length+" and cycles="+cycles);
	}

}
