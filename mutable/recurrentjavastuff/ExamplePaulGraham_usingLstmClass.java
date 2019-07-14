package mutable.recurrentjavastuff;
import static mutable.util.Lg.*;
import java.io.File;
import java.util.Random;
import java.util.function.Supplier;

import immutable.lstm.MatD;
import immutable.lstm.RjLearnStep;
import immutable.lstm.RjLstm;
import immutable.lstm.RjNodesState;
import immutable.lstm.RjPredictStep;
import immutable.recurrentjava.flop.unary.LinearUnit;
import immutable.util.MathUtil;
import mutable.recurrentjava.model.LinearLayer;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.recurrentjava.util.NeuralNetworkHelper;
import mutable.util.Files;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.recurrentjava.RjOptions;
import mutable.recurrentjava.datasets.TextGeneration;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.loss.LossSoftmax;

public class ExamplePaulGraham_usingLstmClass{
	
	
	/*
	DONE, WORKS[
		TODO cast to float in matmul to test if recurrentjava works with that precision
		else will in opencl need to use 2 ints per scalar during matmul etc.
	]
	
	DONE, WORKS WITH LOWER LEARNRATE USING learnTexts_testDelayedUpdateOfWeights[
		TODO in nonparallel recurrentjava delay the changing of weights
		until 5 of them are done like a batch but actually sequentially,
		to verify parallelSize/parallelIndex is a workable theory,
		and I'm very confident it will work since lstm remembers older things learned
		and cuz simialar works in RBM.
	]
	
	TODO use deterministic strong Random to create 2 RjLstms and run them
	side by side to verify that except roundoff its the same data.
	...
	This is hard to do with stateful code. I might need occamsfuncer.
	
	TODO turn back on RjOptions.opencl
	
	TODO turn off on RjOptions.testDelayedUpdateOfWeights
	
	TODO queue many opencl kernels for recurrentjava optimization instead of 1 kernel per OpenclUtil call
	*/
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/** FIXME all these numbers are wrong. 0 dev. too big ave. no bias.
	It did this 2019-5-9 after the upgrade to parallelSize/parallelIndex,
	so the upgrade must be incomplete.
	[RjLstm
		wix[MatD 200x27 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wih[MatD 200x200 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wfx[MatD 200x27 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wfh[MatD 200x200 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wox[MatD 200x27 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		woh[MatD 200x200 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wcx[MatD 200x27 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		wch[MatD 200x200 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		outFeedforwardW[MatD 27x200 ave112.0_dev0.0 stepCacheAve0.0_dev0.0]
		bi[MatD 200x1 ave0.0_dev0.0 stepCacheAve0.0_dev0.0]
		bf[MatD 200x1 ave0.0_dev0.0 stepCacheAve0.0_dev0.0]
		bo[MatD 200x1 ave0.0_dev0.0 stepCacheAve0.0_dev0.0]
		bc[MatD 200x1 ave0.0_dev0.0 stepCacheAve0.0_dev0.0]
		outFeedforwardB[MatD 27x1 ave0.0_dev0.0 stepCacheAve0.0_dev0.0]
		nmhpnull
	]
	This is without parallel code:
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
	*/
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*TODO before continuing ChuasCircuit, verify Lstm.java works here.
	
	That layer is created like this:
	int bottleneckDimension = 10;
	new LinearLayer(inputDimension, bottleneckDimension, initParamsStdDev, rng)
	
	Alternatively I could check if it works without bottleneckDimension
	and if that works then plug the onehot directly into Lstm.java.
	*/
	
	static char indexToNormedChar(int index){
		if(index < 0 || index >= charIndexs) throw new IndexOutOfBoundsException(""+index);
		if(index==26) return ' ';
		return (char)('a'+index);
	}
	
	static int charIndexs = 27;
	
	static int charIndex(char c){
		char low = Character.toLowerCase(c);
		if('a'<=low && low<='z') return low-'a';
		return 26; //all nonletters
	}
	
	static double[] charToOnehot(char c){
		double[] ret = new double[charIndexs];
		ret[charIndex(c)] = 1;
		return ret;
	}
	
	static char onehotToChar(double[] onehot){
		return indexToNormedChar(MathUtil.indexOfMax(onehot));
	}
	
	/** adds 0s at end or removes from end *
	static double[] setSize(double[] d, int newSize){
		double[] ret = new double[newSize];
		System.arraycopy(d, 0, ret, 0, Math.min(d.length, newSize));
		return ret;
	}*/
	
	static RjLearnStep[] textToLearnSteps(String text){
		RjLearnStep[] ret = new RjLearnStep[text.length()-1];
		for(int i=0; i<ret.length; i++){
			ret[i] = new RjLearnStep(
				MatD.rjInVec(charToOnehot(text.charAt(i))),
				MatD.rjOutVec(charToOnehot(text.charAt(i+1)))
			);
		}
		return ret;
	}
	
	static RjLearnStep[] textsToLearnStepsEachParallel(String[] texts){
		int parallelSize = texts.length;
		int singleSize = charIndexs;
		if(singleSize < 2) throw new Error("Must be at least 2 time steps, and input and its output");
		for(int i=1; i<texts.length; i++){
			if(texts[0].length() != texts[i].length()) throw new Error(
				"In a batch, all parallel sequences must be same length");
		}
		RjLearnStep[] ret = new RjLearnStep[texts[0].length()-1];
		for(int i=0; i<ret.length; i++){
			//ret[i] = new RjLearnStep(
			//	MatD.rjInVec(charToOnehot(text.charAt(i))),
			//	MatD.rjOutVec(charToOnehot(text.charAt(i+1)))
			//);
			double[] ins = new double[singleSize*parallelSize]; //MatD.rjInVecs uses singleInSize 1
			double[] outs = new double[ins.length]; //MatD.rjOutVec uses singleOutSize 1
			for(int p=0; p<parallelSize; p++){
				double[] in = charToOnehot(texts[p].charAt(i));
				double[] out = charToOnehot(texts[p].charAt(i+1)); //TODO optimize: equals last in
				System.arraycopy(in, 0, ins, p*in.length, in.length); //copy col
				System.arraycopy(out, 0, outs, p*out.length, out.length); //copy col
			}
			ret[i] = new RjLearnStep(
				//TODO change order of dims in whole recurrentjava so parallelSize is first?
				//Would that break anything?
				new MatD(singleSize, ins),
				new MatD(singleSize, outs)
			);
		}
		return ret;
	}
	
	static String[] padToSameLen(String[] texts, Supplier<Character> padder){
		String[] ret = texts.clone();
		int maxLen = 0;
		for(int i=0; i<ret.length; i++){
			maxLen = Math.max(maxLen, ret[i].length());
		}
		for(int i=0; i<ret.length; i++){
			while(ret[i].length() < maxLen) ret[i] += padder.get(); //slow way to pad but not the bottleneck
		}
		return ret;
	}
	
	/** texts must be same len.
	FIXME 2019-5-7-940a I'm unsure if recurrentjava already supports this, since all its Matrix which
	it uses as 1d (1 of the dims is size 1) are actually 2d.
	*/
	static RjLstm learnTexts(RjLstm lstm, int repeat, double learnRate, String[] texts){
		return lstm.learn(repeat, learnRate, LinearUnit.instance, new LossSoftmax(),
			textsToLearnStepsEachParallel(texts));
	}
	
	static RjLstm learnText(RjLstm lstm, int repeat, double learnRate, String text){
		return lstm.learn(repeat, learnRate, LinearUnit.instance, new LossSoftmax(), textToLearnSteps(text));
	}
	
	static RjLstm learnTexts_testDelayedUpdateOfWeights(
			RjLstm lstm, int repeat, double learnRate, String[] texts){
		//actually parallel things are the nonparallel form sequentially in testDelayedUpdateOfWeights
		int parallelSize = texts.length;
		RjLearnStep[][] sequences = new RjLearnStep[parallelSize][];
		for(int p=0; p<parallelSize; p++){
			sequences[p] = textToLearnSteps(texts[p]);
		}
		return lstm.learn_testDelayedUpdateOfWeights(
			repeat, learnRate, LinearUnit.instance, new LossSoftmax(), sequences);
	}
	
	/** Recurrentjava's example uses an endOfText char and a max len,
	but since im just using this as a testcase before I move on to other things
	Ill do it the easier way.
	*/
	static String predictTextIncludingAndAfter(String prefix, RjLstm lstm, int maxSuffixLen){
		int parallelSize = 1;
		RjNodesState n =  lstm.newEmptyNodesStates(parallelSize);
		char lastOutChar = ' ';
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<prefix.length()+maxSuffixLen; i++){
			char inChar = i<prefix.length() ? prefix.charAt(i) : lastOutChar;
			RjPredictStep in = new RjPredictStep(true, n, MatD.rjInVec(charToOnehot(inChar)));
			RjPredictStep out = lstm.predict(in, LinearUnit.instance);
			n = out.context;
			lastOutChar = onehotToChar(out.vecs.a);
			//if(sb.length()==0) sb.append(inChar);
			sb.append(sb.length() < prefix.length() ? prefix.charAt(i) : lastOutChar);
		}
		return sb.toString();
	}
	
	public static void main(String[] args) throws Exception{
		RjLstm.test();
		
		//String textSource = "PaulGrahamSmall";
		String textSource = "PaulGrahamSmallAndCutToSameLen";
		String fileContent = Text.bytesToString(
			Files.readFileOrInternalRel("data/recurrentjava/"+textSource+".txt"));
		String[] lines = Text.lines(fileContent);
		for(String line : lines) lg("LINE: "+line);
		int ins = charIndexs, nodes = 200, outs = ins;
		lg("Making LSTM...");
		double initParamsStdDev = .08;
		//RjLstm lstm = new RjLstm(ins, nodes, outs, ()->(initParamsStdDev*Rand.strongRand.nextGaussian()));
		RjLstm lstm = new RjLstm(ins, nodes, outs, ()->(initParamsStdDev*Rand.strongRand.nextGaussian()));
		lg("LSTM STARTS AS:\r\n"+lstm);
		double learnRate = .001;
		//double learnRate = .0002;
		//TODO opencl optimize so each LearnStep does many vecs in parallel
		int repeatPerLearnCycle = 1;
		//int repeatPerLearnCycle = 1;
		int learnCycles = 1000/repeatPerLearnCycle;
		lg("Learning...");
		for(int i=0; i<learnCycles; i++){
			//if(RjLstm.learnParallel){
				lstm = learnTexts(
					lstm,
					repeatPerLearnCycle,
					learnRate/50, //FIXME
					padToSameLen(lines, ()->'p')
				);
				/*lg("TESTING PARALLEL CODE ON 1 AT A TIME. SMALL STEPS TOWARD OPENCL...");
				for(String line : lines){
					lstm = learnTexts(
						lstm,
						repeatPerLearnCycle,
						learnRate,
						new String[]{ line }
					);
				}*/
				/*lg("learnTexts_testDelayedUpdateOfWeights...");
				lstm = learnTexts_testDelayedUpdateOfWeights(
					lstm,
					repeatPerLearnCycle,
					learnRate/5, //FIXME
					lines
				);*/
				/*lg("learnTexts_testDelayedUpdateOfWeights 1 at a time which is the wrong test...");
				for(String line : lines){
					lstm = learnTexts_testDelayedUpdateOfWeights(
						lstm,
						repeatPerLearnCycle,
						learnRate,
						new String[]{ line }
					);
				}*/
				
			/*}else{
				for(String line : lines){
					lstm = learnText(lstm, repeatPerLearnCycle, learnRate, line);
				}
			}
			*/
			
			/* After learning a 5 sentence dataset well:
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
			*/
			lg(lstm);
			//if(i%10==0){
				lg("Learn cycle "+(i+1)+"/"+learnCycles+" done");
				for(String line : lines){
					String in = line.substring(0,line.length()/2);
					lg("");
					lg(line+" (correct)");
					lg(in+": (predict...)");
					String out = predictTextIncludingAndAfter(in, lstm, line.length()-in.length()); //not predicting how long it is
					lg(out);
				}
			//}
		}
		lg("done.");
	}
}
