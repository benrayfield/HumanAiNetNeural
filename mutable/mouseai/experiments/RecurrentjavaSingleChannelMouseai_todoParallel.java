/** Ben F Rayfield offers this software opensource MIT license */
package mutable.mouseai.experiments;
import static mutable.util.Lg.*;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Random;
import java.util.TreeMap;
import java.util.WeakHashMap;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.ScrollPaneConstants;

import bsh.Interpreter;
import bsh.NameSpace;
import bsh.Primitive;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import immutable.occamsjsonds.JsonDS;
import immutable.recurrentjava.flop.unary.LinearUnit;
import immutable.recurrentjava.flop.unary.SigmoidUnit;
import immutable.rnn.RnnParams;
import immutable.util.MathUtil;
import immutable.util.Text;
import mutable.chuascircuit.ChuasCircuit;
import mutable.chuascircuit.ChuasCircuitLearnedByRecurrentjava;
import mutable.listweb.ListwebRoot;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.mouseai.UnidimView;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.loss.LossSoftmax;
import mutable.recurrentjava.loss.LossSumOfSquares;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixStat;
import mutable.recurrentjava.model.FeedForwardLayer;
import mutable.recurrentjava.model.GruLayer;
import mutable.recurrentjava.model.LstmLayer;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.model.NeuralNetwork;
import mutable.recurrentjava.trainer.Trainer;
import mutable.recurrentjava.util.NeuralNetworkHelper;
import mutable.uitool.Uitool;
import mutable.util.ColorUtil;
import mutable.util.ui.ColorAtFraction;
import mutable.util.ui.ColorFlowFromCenterColumn;
import mutable.util.ui.ScreenUtil;

/** like RecurrentjavaTimeWindowMouseai except theres only 1 input and 1 output,
so inTheory it needs much less lstm nodes to learn a simple pattern.
*/
public class RecurrentjavaSingleChannelMouseai_todoParallel extends Uitool{
	
	/** should be true but false helps find learning algorithm problems */
	boolean randomizeNodeStatesInReset = false; //FIXME
	//boolean randomizeNodeStatesInReset = true;
	
	public static final float defaultRandomizeOutAve = 0;
	//public static final double defaultRandomizeOutDev = 2.5;
	public static final float defaultRandomizeOutDev = .2f;
	public static final float defaultRandomizeMemAve = 0;
	public static final float defaultRandomizeMemDev = defaultRandomizeOutAve*3;
	
	public boolean setInputToAll0s = false; //FIXME is this hooked in yet?
	
	//double learningRate = 0.03;
	//double learningRate = 0.001;
	//double learningRate = 0.00001;
	//double learningRate = 0.003;
	//public double learningRate = 0.00001;
	//public double learningRate = 0.0005;
	//public double learningRate = 0.001;
	//public double learningRate = 0.001;
	//public double learningRate = 0.0007;
	//public double learningRate = 0.00007;
	//public double learningRate = .0001;
	//public double learningRate = .0003;
	/*use rnnParams().learnRate instead
	public double learnRate(){
		return d("learnRate", .0003);
	}*/
	
	//double maxRadius = 5.2;
	//double maxRadius = 2.6;
	//double maxRadius = 4;
	//double maxRadius = 2.6;
	//double maxRadius = 4.2;
	float maxRadius = 3.2f;
	
	protected Model neuralnet;
	
	/** swaps between 2 copies of mutable.recurrentjava.model.GruLayer.context,
	one for training and one for prediction. The prediction one is never reset
	and is used only during prediction but a different one is used during training.
	Its swapped in this.neuralnet many times per second.
	*/
	protected Matrix swappedState;
	
	/** true if swappedState is the prediction Matrix and neuralnet contains the training matrix,
	else that but swapped.
	*/
	protected boolean swappedStateIsPrediction = true;
	
	/** newest are at highest index. TODO when list gets too big, remove low half of it.
	Keeps adding to this.
	*/
	protected final List<Float> correctOuts = new ArrayList();
	
	/** never reset prediction state, but reset training state at start of each DataSequence.
	Training creates its own Graphs.
	FIXME?? is this graph being used at all? I'm not backproping in it. swappedState is probably all thats needed
	and could use a new Graph every prediction to get the same effect???
	*/
	protected final Graph predictionGraph = new CpuGraph(false);
	//protected final Graph predictionGraph = new Graph(false);
	
	//public final double scaleInputSpeedButNotThinkTime = .5;
	public final float scaleInputSpeedButNotThinkTime = 1;
	
	//public final int timeWindowSize = 20, chaostimeCycles = 5, hiddens = 70;
	//public final int timeWindowSize = 20, chaostimeCycles = 10, hiddens = 70;
	//public final int timeWindowSize = 100, chaostimeCycles = 30, hiddens = 200;
	//public final int timeWindowSize = 200, chaostimeCycles = 50, hiddens = 200;
	//public final int timeWindowSize = 100, chaostimeCycles = 30, hiddens = 300;
	//public final int chaostimeCycles = 20, hiddens = 100;
	//public final int chaostimeCycles = 20, hiddens = 200;
	//public final int chaostimeCycles = 20, hiddens = 100;
	//public final int chaostimeCycles = 30, hiddens = 50;
	//public final int chaostimeCycles = 30, hiddens = 100;
	
	//public final int chaostimeCycles = (int)(30/scaleInputSpeedButNotThinkTime);
	public final int chaostimeCycles = 20; //FIXME
	
	//TODO make this a uitool param
	//public final int hiddens = 180;
	//public final int hiddens = 300;
	//public final int hiddens = 20; //learns 5 dataseqs from chuascircuit near accurately enough
	public final int hiddens = 40;
	
	//TODO less influence the farther back in both dimensions of timewindow,
	//the List and the float[]s in the List.
	//int cycles = 50;
	//int cycles = chaostimeCycles*10;
	//int cycles = 100; //FIXME chaostimeCycles*30?
	//int cycles = 20; //FIXME chaostimeCycles*30?
	//int cycles = 20; //FIXME chaostimeCycles*30?
	//int cycles = 150;
	//final int dataseqSizeToLearnAtOnce = chaostimeCycles*4;
	final int dataseqSizeToLearnAtOnce = (int)(210/scaleInputSpeedButNotThinkTime);
	
	/** indexs 0-(timeWindowSize*2-1) are mouseYFraction and prediction of it.
	The rest are hidden neuralnet node outputs.
	Evens are observation. Odds are prediction.
	Color prediction white and observation before chaostime is green and observation after chaostime is blue.
	Where the white matches the blue its predicting what it wasnt told,
	which will in mouseai be displayed to user as an alternate path mouse could have moved that was predicted
	from chaostime back.
	Where the white matches the green its predicting what it was told,
	so thats easy but a problem if it doesnt predict well.
	<br><br>
	FIXME where do predictions go.
	FIXME only give it data from chaostime back to predict from
	*/
	protected final float[] display;
	
	protected final float[] redMult, greenMult, blueMult;
	
	boolean resetPredictionStateNextTime;
	
	//public String dataSource;
	
	protected List<UnidimView> learnseqs = new ArrayList();
	
	protected ColorFlowFromCenterColumn gameArea;
	
	private transient boolean isClosing;
	
	protected JTextArea messageToUser;

	/** written by Uitool.accept(Object) where that Object is a String
	whose first token is uitool:thisclassname and the rest is beanshell code
	and after it runs its namespace (only including values that are
	Number (use doubleValue) or String (maybe will allow more types later)
	are put into this map, which are neuralnet params and other options.
	See the defs in mindmap for examples of such strings containing beanshell code.
	*/
	protected NavigableMap<String,Object> params = Collections.emptyNavigableMap();
	
	/** get double else elseDefault from params */
	public double d(String name, double elseDefault){
		Object o = (Double) params.get(name);
		return o instanceof Double ? (Double)o : elseDefault;
	}
	
	/** else 0 */
	public double d(String name){
		return d(name, 0.);
	}
	
	public float f(String name, float elseDefault){
		return (float)d(name,(double)elseDefault);
	}
	
	/** else 0 */
	public float f(String name){
		return f(name, 0f);
	}
	
	public boolean exists(String name){
		return params.containsKey(name);
	}
	
	public RnnParams rnnParams(){
		RnnParams p = new RnnParams();
		if(exists("learnRate")) p = p.learnRate(f("learnRate"));
		if(exists("rjTrainerDecayRate")) p = p.rjTrainerDecayRate(f("rjTrainerDecayRate"));
		if(exists("rjTrainerSmoothEpsilon")) p = p.rjTrainerSmoothEpsilon(f("rjTrainerSmoothEpsilon"));
		if(exists("rjTrainerGradientClipValue")) p = p.rjTrainerGradientClipValue(f("rjTrainerGradientClipValue"));
		if(exists("rjTrainerRegularization")) p = p.rjTrainerRegularization(f("rjTrainerRegularization"));
		return p;
	}
	
	public String dataSource(){
		String d = (String) params.get("dataSource");
		//return d!=null ? d : "1dViewOfChuasCircuit";
		
		//FIXME I'm hardcoding this cuz Uitool.accept isnt called
		//until after my constructor, but UnidimViews (which contain the data)
		//are created in constructor,
		//so for not it appears that only 1 dataset is supported at a time
		//return d!=null ? d : "mm:mouseY2019-12-3-210p";
		return d!=null ? d : "mm:mouseY2019-12-3-3p";
	}
	
	public RecurrentjavaSingleChannelMouseai_todoParallel(){
		setLayout(new GridLayout(0,1));
		gameArea = new ColorFlowFromCenterColumn(
			new ColorAtFraction(){
				public int colorAtFraction(double fraction, Map params){
					return -1; //white
				}
			},
			new HashMap(),
			2
		){
			double lastEventWhen = Time.now();
			
			public void mouseMoved(MouseEvent e){
				super.mouseMoved(e);
				if(dataSource().equals("mouseYFraction")){
					double now = Time.now();
					double dt = now-lastEventWhen;
					//if(dt > 1./50){
					if(dt > 1./30){
						lastEventWhen = now;
						readUiAndThinkAndUpdateDisplayArray();
						repaint();
					}
				}
			}
			
			public void paint(Graphics g){
				super.paint(g);
				int radius = 3;
				int centerX = getWidth()/2;
				int heightMult = 5;
				
				g.setColor(Color.black);
				g.fillRect(0, 0, 100, 70);
				
				g.setColor(inputColor);
				g.drawString("in", 20, 20);
				//float input = getCorrectOutAtRelCycle(0);
				//int pixelHeightOfMouseY = -(int)(mouseYFraction*getHeight())+getHeight()/2;
				int pixelHeightOfInput = (int)(in*getHeight());
				g.fillOval(centerX-radius, pixelHeightOfInput-radius*heightMult, radius*2, radius*2*heightMult);
				//float mouseYFraction = getInputAtRelCycle(0);
				//int pixelHeightOfPredictMouseYMinusMouseY = -(int)((predictMouseYFraction-mouseYFraction)*getHeight())+getHeight()/2;
				//g.fillOval(centerX-radius, pixelHeightOfPredictMouseYMinusMouseY-radius, radius*2, radius*2);
				
				g.setColor(correctOutColor);
				g.drawString("correctOut", 20, 40);
				int pixelHeightOfCorrectOuteY = (int)(correctOut*getHeight());
				//g.drawLine(0, pixelHeightOfPredictMouseY, getWidth(), pixelHeightOfPredictMouseY);
				g.fillOval(centerX-radius, pixelHeightOfCorrectOuteY-radius*heightMult, radius*2, radius*2*heightMult);
				
				g.setColor(observedOutColor);
				g.drawString("observedOut", 20, 60);
				int pixelHeightOfObservedOut = (int)(observedOut*getHeight());
				//g.drawLine(0, pixelHeightOfPredictMouseY, getWidth(), pixelHeightOfPredictMouseY);
				g.fillOval(centerX-radius, pixelHeightOfObservedOut-radius*heightMult, radius*2, radius*2*heightMult);
				
			}
		};
		//this.dataSource = dataSource;
		display = new float[hiddens];
		redMult = new float[hiddens];
		greenMult = new float[hiddens];
		blueMult = new float[hiddens];
		gameArea.setPainter(new ColorAtFraction(){
			public int colorAtFraction(double fraction, Map params){
				//return 0xff000000|Rand.strongRand.nextInt(0x1000000);
				int index = (int)MathUtil.holdInRange(0, fraction*display.length, display.length-1);
				//float s = (float)MathUtil.sigmoid(display[index]*2-1);
				float s = MathUtil.holdInRange(0, display[index], 1);
				return ColorUtil.color(redMult[index]*s, greenMult[index]*s, blueMult[index]*s);
			}
		});
		Arrays.fill(redMult, .7f);
		Arrays.fill(greenMult, .7f);
		Arrays.fill(blueMult, .7f);
		/*for(int i=0; i<display.length; i++){
			if(i<timeWindowSize*2){
				if((i&1)==0){ //observation
					if(i<timeWindowSize*2-chaostimeCycles*2){ //older than chaostime, neuralnet can see it
						redMult[i] = 0;
						greenMult[i] = .7f;
						blueMult[i] = 0;
					}else{ //newer than chaostime, neuralnet cant see it except in older observations
						redMult[i] = .2f;
						greenMult[i] = .2f;
						blueMult[i] = 1;
					}
				}else{ //prediction
					redMult[i] = .7f;
					greenMult[i] = .7f;
					blueMult[i] = .7f;
				}
			}else{ //neuralnet node outputs
				redMult[i] = .4f;
				greenMult[i] = .4f;
				blueMult[i] = .4f;
			}
		}*/
		
		//double initParamsStdDev = 0.08;
		//double initParamsStdDev = 0.01;
		//double initParamsStdDev = 0.015;
		//double initParamsStdDev = 0.0015;
		//double initParamsStdDev = 0.015;
		//double initParamsStdDev = 0.5;
		//double initParamsStdDev = 1.5;
		//double initParamsStdDev = 0.4;
		//double initParamsStdDev = .5;
		float initParamsStdDev = .7f;
		
		//int ins = timeWindowSize-chaostimeCycles;
		int ins = 1;
		//int outs = timeWindowSize;
		int outs = 1;
		/*neuralnet = NeuralNetworkHelper.makeGru(
			ins,
			hiddens, 1, 
			outs, new SigmoidUnit(), 
			initParamsStdDev, Rand.strongRand);
		*/
		//int parallelSize = 20;
		int parallelSize = 5;
		//int parallelSize = 12;
		//int parallelSize = 1;
		//int parallelSize = 2;
		neuralnet = NeuralNetworkHelper.makeGru(
			parallelSize,
			ins,
			hiddens, 1, 
			new int[]{hiddens, outs}, new SigmoidUnit(), 
			initParamsStdDev, Rand.strongRand);
		neuralnet.resetState(); //creates Matrix
		swap();
		neuralnet.resetState(); //creates Matrix
		//JPanel topPanel = new JPanel(new BorderLayout());
		messageToUser = new JTextArea();
		messageToUser.setEditable(false);
		messageToUser.setWrapStyleWord(true);
		messageToUser.setLineWrap(true);
		//topPanel.add(messageToUser, BorderLayout.NORTH);
		//topPanel.add(cf, BorderLayout.CENTER);
		//add(topPanel);
		JPanel learnseqsPanel = new JPanel(new GridLayout(0,1));
		int learnseqsToCreate = parallelSize; //TODO much more, but will only display this many
		for(int i=0; i<learnseqsToCreate; i++){
			//UnidimView u = new UnidimView(dataseqSizeToLearnAtOnce);
			UnidimView u = eatIOTimestepsIntoUnidimview(); //FIXME if input type is mouse* this wont work, but works for chuascircuit
			learnseqs.add(u);
			learnseqsPanel.add(u);
		}
		add(messageToUser);
		add(gameArea);
		int v = ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS;
		int h = ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER;
		add(new JScrollPane(learnseqsPanel, v, h));
		revalidate();
		
		new Thread(()->{
			if(true /*dataSource().equals("1dViewOfChuasCircuit")*/){
				if(isOnlyUnidimviews){
					swap();
				}
				while(!isClosing){ //FIXME this continues even after the window is closed so not researchmindmap compatible
					lg("RecurrentjavaSimpleMouseai looping");
					if(isOnlyUnidimviews){
						//for(UnidimView u : learnseqs){
						//	if(!isClosing){
						//		learnAndPredictAndUpdateOutputAndRepaint(u);
						//	}
						//}
						
						if(!isClosing){
							parallelLearnAndPredictAndUpdateOutputAndRepaint(learnseqs.toArray(new UnidimView[0]));
						}
					}else{
						if(!isClosing){
							readUiAndThinkAndUpdateDisplayArray();
						}
						if(!isClosing){
							gameArea.repaint(); //doesnt erase o.cf's last painted state like o.repaint() does.
						}
						//ColorFlowFromCenterColumn is designed to slide over its last painted state, so commentedout this to not erase it
						//repaint();
					}
					
					if(!isClosing){
						//Thread.yield();
						//Time.sleepNoThrow(1./30);
						Time.sleepNoThrow(1./400);
					}
				}
			}
		}).start();
	}
	
	/** uitool gives uitool:myclassname then beanshell code, which I run then grab all
	the vars whose values are Number and put them in a map of String to Double which I use
	as neuralnet params and other options. This uses a new beanshell interpreter each call
	since whats typed into ResearchPanel (which becomes the String param here)
	is supposed to be reusable when stored in a researchmindmap def, not stateful.
	The beanshell code is for easily changing multiple things
	derived from fewer things that you actually change,
	like RecurrentJava LstmLayer and GruLayer have multiple Matrix
	and you might want to change maxRadius on each matrix separately
	but by a ratio or other code, and only change the scaling of all those
	in a single var in the beanshell code, or other ways of grouping things.
	*/
	public void accept(Object o){
		String s = (String)o;
		String uitoolColonConcatMyClassName = Text.firstWhitespaceDelimitedToken(s);
		String bshCode = s.substring(uitoolColonConcatMyClassName.length()).trim();
		StringBuilder msg = new StringBuilder();
		try{
			Interpreter bsh = new Interpreter(new InputStreamReader(System.in), System.out, System.err, true);
			lg("Evaling beanshell code["+bshCode+"]");
			bsh.eval("int testFunc(int x){ return x; }"); //cuz getAllNames meeds at least 1 function to exist else throws
			bsh.eval(bshCode);
			NameSpace ns = bsh.getNameSpace();
			NavigableMap map = new TreeMap();
			String[] varNames = ns.getAllNames();
			Arrays.sort(varNames);
			for(String varName : varNames){
				Object varVal = Primitive.unwrap(ns.getVariable(varName, true)); //works if primitive or nonprimitive
				if(varVal instanceof Number){
					map.put(varName, ((Number)varVal).doubleValue());
				}else if(varVal instanceof String){
					map.put(varName, varVal);
				}
			}
			this.params = Collections.unmodifiableNavigableMap(map);
			for(Map.Entry entry : this.params.entrySet()){
				msg.append(entry.getKey()).append(" = ")
					.append(JsonDS.jsonString(entry.getValue())).append(";").append(Text.n);
			}
			lg(getClass().getName()+".param = "+this.params);
		}catch(Throwable t){
			msg.setLength(0);
			msg.append(Text.toString(t));
		}
		messageToUser.setText(msg.toString().trim());
	}
	
	public static final Color inputColor = new Color(.7f, .2f, .2f);
	
	public static final  Color correctOutColor = new Color(.6f, .6f, 1f);
	
	public static final  Color observedOutColor = new Color(0f, .8f, 0f);
	
	/** changes swappedStateIsPrediction and swappedState */
	public void swap(){
		Matrix temp = ((GruLayer)((NeuralNetwork)neuralnet).layers.get(0)).context;
		((GruLayer)((NeuralNetwork)neuralnet).layers.get(0)).context = swappedState;
		swappedState = temp;
		swappedStateIsPrediction = !swappedStateIsPrediction;
	}
	
	/** param is 0 for newest input, 1 for second newest, and so on. Returns .5 if farther back than exists */
	public float getCorrectOutAtRelCycle(int cyclesBack){
		int j = correctOuts.size()-1-cyclesBack;
		return 0<=j && j<correctOuts.size() ? correctOuts.get(j) : .5f;
	}
	
	public float[] getCorrectOutRangeAtRelCycles(int cyclesBackStart, int len){
		float[] f = new float[len];
		for(int i=0; i<len; i++){
			f[i] = getCorrectOutAtRelCycle(cyclesBackStart-i);
		}
		return f;
	}
	
	/** FIXME should the input come from startCyclesBack or from startCyclesBack-chaosTimeCycles? */
	public DataSequence trainingData(int startCyclesBack, int cycles){
		//FIXME verify this aligns with the prediction code, both are chaostimeCycles apart
		List<DataStep> list = new ArrayList();
		for(int i=0; i<cycles; i++){
			int outIndex = startCyclesBack-i;
			float correctOut = getCorrectOutAtRelCycle(outIndex);
			float in = getCorrectOutAtRelCycle(outIndex+chaostimeCycles);
			list.add(new DataStep(new float[]{in}, new float[]{correctOut}));
		}
		/*
		//int ins = timeWindowSize-chaostimeCycles;
		//int outs = timeWindowSize;
		int ins = 1, outs = 1;
		//dont train on newest chaostime or let neuralnet get it as inputs, but still neuralnet must predict it as outputs
		//int startHowFarBack = cycles+outs+chaostimeCycles;
		for(int i=0; i<cycles; i++){
			int startAtCyclesBack = startCyclesBack-i;
			list.add(new DataStep(
				MathUtil.toDoubles(getInputRangeAtRelCycles(startAtCyclesBack, ins)),
				MathUtil.toDoubles(getInputRangeAtRelCycles(startAtCyclesBack, outs))
			));
		}*/
		return new DataSequence(list);
	}
	
	public static Matrix toMatrix(float[] f){
		return new Matrix(f.clone());
	}
	
	public static Matrix toMatrix(double[] f){
		return new Matrix(MathUtil.toFloats(f)); //cuz use array as immutable
	}
	
	public static float[] toFloats(Matrix m){
		return m.mem("w").toFloatArray();
	}
	
	/*public float[] getNeuralnetOutputs(){
		((FeedForwardLayer)((NeuralNetwork)neuralnet).layers.get(1)).
	}*/
	
	
	//prediction of now's mouseYFraction by weights and state that have never seen mouseY
	//in time range now-chaostime to now.
	//TODO verify it cant cheat, cuz I might have code that wrong.
	protected float observedOut = 0; //prediction
	/** what observedOut should learn to be */
	protected float correctOut = 0;
	/** copy of correctOut from chaosTimeCycles in the past (so cant cheat by copying correctOut) */
	protected float in = 0;
	
	public ChuasCircuit chua = ChuasCircuitLearnedByRecurrentjava.randomChuasCircuitStartState(Rand.strongRand);
	
	void randomizeGruOrLstmNodeStates(Model neuralnet){
		randomizeGruOrLstmNodeStates(neuralnet, defaultRandomizeOutAve, defaultRandomizeOutDev,
			defaultRandomizeMemAve, defaultRandomizeMemDev);
	}
	
	/** Only uses memAve and memDev in LstmLayers.
	GruLayer and LstmLayer both have Matrix context so use outAve and outDev.
	*/
	void randomizeGruOrLstmNodeStates(Model neuralnet, float outAve, float outDev, float memAve, float memDev){
		List<Model> layers = ((NeuralNetwork)neuralnet).layers;
		for(Model layer : layers){
			if(layer instanceof GruLayer){
				randomize(((GruLayer)layer).context, outAve, outDev);
			}else if(layer instanceof LstmLayer){
				randomize(((LstmLayer)layer).cellContext, outAve, outDev);
				randomize(((LstmLayer)layer).hiddenContext, memAve, memDev);
			}
		}
	}
	
	void randomize(Matrix m, float ave, float dev){
		Random rand = Rand.strongRand;
		FSyMem w = m.mem("w");
		for(int i=0; i<m.size; i++){
			w.put(i, ave+(float)(rand.nextGaussian()*dev));
		}
	}
	
	
	protected void parallelLearnAndPredictAndUpdateOutputAndRepaint(UnidimView... u){
		//FIXME this isnt parallel yet
		RnnParams p = rnnParams();
		DataSequence ds[] = new DataSequence[u.length];
		for(int i=0; i<u.length; i++) ds[i] = u[i].asDataSequence();
		if(!swappedStateIsPrediction) throw new Error("training matrix should be in neuralnet but is in swappedState");
		
		//FIXME merge duplicate code between readUiAndThinkAndUpdateDisplayArray and learnAndPredictAndUpdateOutputAndRepaint
		boolean applyTraining = true;
		Loss lossTraining = new LossSumOfSquares(); //for waves such as mouse
		Loss lossReporting = lossTraining;
		int cycle[] = new int[u.length];
		Consumer<Matrix> copyEachDoubleFromMatrixToUnidimview = (Matrix m)->{ //float
			//if(m.w.length != 1) throw new Error("output matrix size is not 1. Its "+m.w.length);
			//FIXME are rows vs cols reversed?
			for(int i=0; i<u.length; i++){
				u[i].data[UnidimView.OBSERVED_OUT][cycle[i]++] = m.mem("w").get(i);
				u[i].repaint();
			}
		};
		try{
			Consumer<Model> stateResetter = (Model m)->{
				m.resetState();
				if(randomizeNodeStatesInReset) randomizeGruOrLstmNodeStates(m);
			};
			Trainer.pass(p, copyEachDoubleFromMatrixToUnidimview, stateResetter, neuralnet, Arrays.asList(ds), applyTraining, lossTraining, lossReporting);
			norm(neuralnet, maxRadius);
			for(Matrix m : neuralnet.getParameters()){
				MatrixStat s = new MatrixStat(m.buf("w"),m.rows,m.cols);
				//lg("Radius.col[0] of "+System.identityHashCode(m)+" is "+s.radiusPerCol[0]+" and of row[0] is "+s.radiusPerRow[0]);
			}
		}catch(Exception e){
			throw new Error(e);
		}
	}
	
	
	protected void learnAndPredictAndUpdateOutputAndRepaint(UnidimView u){
		RnnParams p = rnnParams();
		DataSequence ds = u.asDataSequence();
		if(!swappedStateIsPrediction) throw new Error("training matrix should be in neuralnet but is in swappedState");
		
		//FIXME merge duplicate code between readUiAndThinkAndUpdateDisplayArray and learnAndPredictAndUpdateOutputAndRepaint
		boolean applyTraining = true;
		Loss lossTraining = new LossSumOfSquares(); //for waves such as mouse
		Loss lossReporting = lossTraining;
		int cycle[] = new int[1];
		Consumer<Matrix> copyOneDoubleFromMatrixToUnidimview = (Matrix m)->{
			if(m.size != 1) throw new Error("output matrix size is not 1. Its "+m.size);
			u.data[UnidimView.OBSERVED_OUT][cycle[0]++] = m.mem("w").get(0);
		};
		try{
			Consumer<Model> stateResetter = (Model m)->{
				m.resetState();
				if(randomizeNodeStatesInReset) randomizeGruOrLstmNodeStates(m);
			};
			Trainer.pass(p, copyOneDoubleFromMatrixToUnidimview, stateResetter, neuralnet, Arrays.asList(ds), applyTraining, lossTraining, lossReporting);
			norm(neuralnet, maxRadius);
			for(Matrix m : neuralnet.getParameters()){
				MatrixStat s = new MatrixStat(m.buf("w"),m.rows,m.cols);
				//lg("Radius.col[0] of "+System.identityHashCode(m)+" is "+s.radiusPerCol[0]+" and of row[0] is "+s.radiusPerRow[0]);
			}
		}catch(Exception e){
			throw new Error(e);
		}
		u.repaint();
	}
	
	int eaten = 0;
	
	Map<String,Object> cacheJsonParse = new WeakHashMap();
	
	public Object jsonParse(String json){
		Object parsed = cacheJsonParse.get(json);
		if(parsed == null){
			parsed = JsonDS.jsonParse(json);
			cacheJsonParse.put(json, parsed);
		}
		return parsed;
	}
	
	/** If its mouseYFraction then have to actually move the mouse before this changes, but for 1dViewOfChuasCircuit that works */
	public float eatAndReturnNextCorrectOut(){
		String dataSource = dataSource();
		float nextCorrectOut;
		if(dataSource.equals("mouseYFraction")){
			nextCorrectOut = (float)(double)gameArea.params.get("mouseYFraction");
		}else if(dataSource.equals("1dViewOfChuasCircuit")){
			for(int i=0; i<200; i++){
				//chua.nextState(.0003); //FIXME what dt param?
				//chua.nextState(.0006); //FIXME what dt param?
				chua.nextState(.0004*scaleInputSpeedButNotThinkTime); //FIXME what dt param?
			}
			//nextInput = (float)(chua.x+chua.y+chua.z)*.02f+.5f;
			//nextInput = (float)(chua.x+chua.y+chua.z)*.1f+.5f;
			nextCorrectOut = (float)(chua.x+chua.y+chua.z)*.07f+.5f;
		}else if(dataSource.startsWith("mm:")){ //json in mindmap node's def such as mm:mouseY2019-12-3-210p
			String json = ListwebRoot.def(dataSource.substring("mm:".length()));
			//List<Double> data = (List<Double>) JsonDS.jsonParse(json); //TODO optimize remember this instead of parsing again every time
			//nextCorrectOut = data.get(eaten%data.size());
			float[] data = (float[]) jsonParse(json);
			nextCorrectOut = data[eaten%data.length];
			eaten++;
		}else{
			throw new Error("Unknown dataSource="+dataSource);
		}
		lg("Next correctOut: "+nextCorrectOut);
		return nextCorrectOut;
	}
	
	public float[] eatAndReturnNextNCorrectOutputs(int n){
		float[] correctOuts = new float[n];
		for(int i=0; i<n; i++) correctOuts[i] = eatAndReturnNextCorrectOut();
		return correctOuts;
	}
	
	public UnidimView eatIOTimestepsIntoUnidimview(){
		UnidimView u = new UnidimView(dataseqSizeToLearnAtOnce);
		float[] correctOuts = eatAndReturnNextNCorrectOutputs(dataseqSizeToLearnAtOnce+chaostimeCycles); 
		for(int i=0; i<dataseqSizeToLearnAtOnce; i++){
			if(!setInputToAll0s) u.data[UnidimView.IN][i] = correctOuts[i];
			u.data[UnidimView.CORRECT_OUT][i] = correctOuts[i+chaostimeCycles];
			//NO, cuz it makes the display scaling too small: leave u.data[UnidimView.OBSERVED_OUT] as 0, to be overwritten each time its predicted and learned together
			//u.data[UnidimView.OBSERVED_OUT][i] = u.data[UnidimView.CORRECT_OUT][i]; //Not true but neither is all 0s, cuz it hasnt output yet
		}
		return u;
	}
	
	protected void readUiAndThinkAndUpdateDisplayArray(){
		
		if(gameArea.params.containsKey("mouseButton"+MouseEvent.BUTTON1)){
			resetPredictionStateNextTime = true;
		}
		
		RnnParams p = rnnParams();
		
		
		
		//Train on random recent time range so retrains on older sometimes
		//to reduce overfitting.
		//Can go off the edge of List inputs but its all constant data past there.
		int startCyclesBack = Rand.strongRand.nextInt(Math.min(correctOuts.size()+1,dataseqSizeToLearnAtOnce*2));
		//no cheating by training on the range now-chaostime to now.
		startCyclesBack = Math.max(startCyclesBack, dataseqSizeToLearnAtOnce+chaostimeCycles);
				
		DataSequence ds = trainingData(startCyclesBack, dataseqSizeToLearnAtOnce);
		swap();
		
		//FIXME merge duplicate code between readUiAndThinkAndUpdateDisplayArray and learnAndPredictAndUpdateOutputAndRepaint
		if(!swappedStateIsPrediction) throw new Error("training matrix should be in neuralnet but is in swappedState");
		boolean applyTraining = true;
		Loss lossTraining = new LossSumOfSquares(); //for waves such as mouse
		//Loss lossTraining = new LossSoftmax(); //for text etc
		Loss lossReporting = lossTraining;
		try{
			Consumer<Model> stateResetter = (Model m)->{
				m.resetState();
				if(randomizeNodeStatesInReset) randomizeGruOrLstmNodeStates(m);
			};
			Trainer.pass(p, Trainer.ignoreMatrix, stateResetter, neuralnet, Arrays.asList(ds), applyTraining, lossTraining, lossReporting);
			norm(neuralnet, maxRadius);
			for(Matrix m : neuralnet.getParameters()){
				MatrixStat s = new MatrixStat(m.buf("w"),m.rows,m.cols);
				//lg("Radius.col[0] of "+System.identityHashCode(m)+" is "+s.radiusPerCol[0]+" and of row[0] is "+s.radiusPerRow[0]);
			}
		}catch(Exception e){
			throw new Error(e);
		}
		swap();
		
		if(swappedStateIsPrediction) throw new Error("prediction matrix should be in neuralnet but is in swappedState");
		if(resetPredictionStateNextTime){
			lg("resetting prediction state");
			//neuralnet.resetState();
			randomizeGruOrLstmNodeStates(neuralnet);
			resetPredictionStateNextTime = false;
		}
		
		//int ins = timeWindowSize-chaostimeCycles;
		//int outs = timeWindowSize;
		int ins = 1, outs = 1;
		//Matrix in = toMatrix(getInputRangeAtRelCycles(outs-1, ins));
		Matrix in = toMatrix(getCorrectOutRangeAtRelCycles(chaostimeCycles, ins));
		this.in = (float)in.mem("w").get(0);
		if(in.size != 1) throw new Error("number of inputs is not 1 so 'this.in = (float)in.w[0];' is the wrong code: "+in.size);
		float[] out;
		try{
			out = toFloats(neuralnet.forward(in, predictionGraph));
		}catch(Exception e){
			throw new Error(e);
		}
		
		/*for(int cyclesBack=0; cyclesBack<timeWindowSize; cyclesBack++){
			display[timeWindowSize*2-2-cyclesBack*2] = getInputAtRelCycle(cyclesBack);
		}*/
		/*for(int i=0; i<timeWindowSize*2-2; i+=2){ //evens are observation, odds are prediction, up to timeWindowSize*2
			//display[i] = display[i+2];
			//int j = inputs.size()-1-(timeWindowSize-1-i/2);
			display[i] = getInputAtRelCycle(timeWindowSize-1-i/2);
		}*/
		
		/*for(int cyclesBack=0; cyclesBack<timeWindowSize; cyclesBack++){ //evens are observation, odds are prediction, up to timeWindowSize*2
			//display[timeWindowSize*2-1-cyclesBack*2] = Rand.strongRand.nextFloat();
			float o = out[timeWindowSize-1-cyclesBack];
			if(cyclesBack==0){
				predictMouseYFraction = o;
			}
			display[timeWindowSize*2-1-cyclesBack*2] = o; //FIXME is this backward?
		}*/
		//prediction = .5f+50f*(out[0]-.5f); //display (and check accuracy) magnified
		observedOut = out[0];
		correctOut = getCorrectOutAtRelCycle(0);
		//System.arraycopy(display, 1, display, 0, timeWindowSize-1);
		//display[timeWindowSize*2-2] = (float)(double)params.get("mouseYFraction"); //written by ColorFlowFromCenterColumn
		correctOuts.add(eatAndReturnNextCorrectOut()); //written by ColorFlowFromCenterColumn
		swap();
		float[] predictionHiddens = toFloats(swappedState);
		swap();
		//if(predictionHiddens.length != display.length-timeWindowSize*2)
		if(predictionHiddens.length != display.length)
			throw new Error("predictionHiddens.length="+predictionHiddens.length);
			//throw new Error("predictionHiddens.length="+predictionHiddens.length
			//	+" display.length-timeWindowSize*2="+(display.length-timeWindowSize*2));
		/*for(int i=timeWindowSize*2; i<display.length; i++){
			//display[i] = Rand.strongRand.nextFloat();
			display[i] = predictionHiddens[i-timeWindowSize*2];
		}*/
		System.arraycopy(predictionHiddens, 0, display, 0, display.length);
		float displayAve = MathUtil.ave(display);
		float displayDev = MathUtil.devGivenAve(displayAve, display);
		if(displayDev != 0) {
			for(int i=0; i<display.length; i++){
				display[i] = .5f + .2f*(display[i]-displayAve)/displayDev;
			}
		}
		
		/*Train it on the timeWindow snapshots from now-10 to now-chaostime.
		Keep a second GruLayer.context thats never reset and replace part of the timewindow
		in it each call of neuralnet.
		Each call of neuralnet, do just 1 neuralnet cycle and get the prediction (especially now to now-chaostime).
		*/
		
		/*I want to instead train it only 1 timestep, and predict 1 timestep (using different Matrix contexts,
		1 chaostime back from the other), BUT if I do that it will probably extremely overfit and be useless.
		Could I add randomness gradually to prevent that?
		NO THATS TOO HARD.
		*/
	}
	
	/** todo different maxRadius per matrix, but get this basic thing working first */
	static void norm(Model model, float maxRadius){
		for(Matrix m : model.getParameters()){
			m.normByMaxRadius(maxRadius, maxRadius);
		}
	}
	
	static final boolean isOnlyUnidimviews = true;
	
	protected void finalize() throws Throwable{
		close();
	}
	
	public void close(){
		isClosing = true; //loop will 
	}
	
	public static void main(String[] args){
		//args[0] is "mouseYFraction" or "1dViewOfChuasCircuit" (may add more possible values later)s
		RecurrentjavaSingleChannelMouseai_todoParallel o = new RecurrentjavaSingleChannelMouseai_todoParallel();
		ScreenUtil.testDisplayWithoutExitOnClose(o);
	}

}
