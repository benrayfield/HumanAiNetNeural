package mutable.uitool.dataset;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.uitool.Uitool;
import mutable.util.ui.ScreenUtil;

/** mouse X and Y are in normal coordinates but views them as 1d waves along a long spiral.
Reads/writes in Serializable float[] of y as even and x as odd index (UPDATE: float[2][*]).
Only records mouse when its moving, and restarts recording when move again.
Mouse is normalized to range 0 to 1.
<br><br>
Spiral coordinates are time and amplitude, similar to radial except it does n turns around a spiral,
like grooves in circular sound record big enough to see.
<br><br>
FIXME timing per 1/64 second (when moving) instead of timing by mouse events.
<br><br>
FIXME the sizes and positions (such as the 2 curves being one tends to be a little farther from center than other)
are arbitrary and will probably look worse on other screen sizes.
*/
public class MouseYRecorder32HzToSerializableFloat1d extends Uitool{
	
	protected double lastTimeMouseMoved = 0;
	
	protected List<Float>[] mouseFractions = new List[]{new ArrayList()};
	
	/** This will be a design pattern in uitools, to have named vars (keys in this map) per uitool instance
	that say where to read andOr write files as rel paths.
	*
	public NavigableMap<String,String> paths;
	*/
	
	/*I want something like copy/paste or drag/drop, except usually I want it to remember from and to data flows,
	so a uitool (defined by a string such as copied from a mindmap def) is the data of that string,
	and a Serializable (byte[]) is another data, and you can drag a Serializable from a file (such as "/data/ser/mouseRecording2019-3abcjksdf.ser")
	into a specific x y point in the JComponent of that instance of that uitool and it will load it,
	or you can drag from such a point to a file and it will save it. I want a ui like I had for iotadesktop occamdesktop etc
	where you drag function onto function to create function, except it has some mutable parts (can add occamsfuncer later).
	That ui is already generalized enough to easily expand to do this.
	I want to use this to record mouse in a uitool, then use a few uitools each to create different kinds of datasets from it
	(such as all scalars, vs 128 384 nhot, vs lstm-like ways, or whatever),
	and to create andOr wrap other datasets such as clibin and 2adjacenttimesofconwaylife and other invarep kinds of testcases.
	And I want a uitool wrapping the learnloop/rbmeditor/slide process of learning and of pushing 1 edlay at a time.
	I want some kind of high dimensional desktop space to put these many objects in,
	like in hypercubewave each keyboard button if you held it while moving mouse you moved that dimension in aftrans,
	maybe I want a desktop like that, or maybe I just want a folderlike structure allowing cycles of folders in folders etc
	or maybe listweb is the best way for that. Remember those keyboard shortcuts I thought of for listweb,
	named things like fly fall flip up down etc... there were about 10-15 (12 it appears? in flipPushPopUpDownInOutGiveGetFlyFallKill)
	of them and allowed the use of listweb even if you are blind, intheory, if it would read the thing to you that is selected
	(flipPushPopUpDownInOutGiveGetFlyFallKill is its mindmapname).
	...
	Occamsfuncer doesnt solve this problem, cuz I could simply store 2 things in 1 immutable data and put that in mindmap def,
	such as one of this ia a uitool and the other is the saved data.
	...
	Consider the immutable model (though not limited to occamsfuncer kind of immutable)
	where you drag immutable onto immutable to create immutable in some cases,
	and in other cases you start with an immutable and use mouse and keyboard etc input as the next inputs
	and that "other cases" process is viewed as uitool. RbmEditor with Slide.java and SlideAction.java etc
	could be immutable, though I had planned to use occamsfuncer map and lazycat instead of those
	(will still do that later, but 1 more gradual redesign between here and there, I could do).
	Each input of a dimension (such as mouseY or keyboardButtonLeft or charA etc) with a value (range -1 to 1,
	0 being what everything is when its not stored) would be a small map (of just 1 key and value) sent into
	the funcer (along with a copy of the funcer itself) that returns a funcer of its next state.
	So you could play a funcer forward like an interactive game, then continue using it like a function again
	in combos with other funcers. For that I would want the iotadesktop-like dragAndDrop working with listweb.
	...
	How would that work, when recording mouse in one function, then creating a 128_384 dataset from that recording,
	then feeding that dataset into rbmeditor to test how well rbm learns it? If I could make funcs at runtime,
	I could easily make a func that automates the dragAndDropping in that process if just given any part I want to
	replace such as a different choice of function that generates the RBM dataset from given mouse recordings,
	or a different RBM learning algorithm, or finding an existing different mouse recording,
	or recording mouse again.
	...
	Yes, I want immutable funcs that can make immutable funcs (eventually including occamsfuncer),
	and I want these stored in a listweblike mindmap, and I want slowFullEconacyc avail.
	...
	I want javaclass graphics available, one of which (later prototype, harder to build it this way)
	will be something that generates an int[][] colorARGB,
	but others will be like *MouseRecorder*.java which displays a spiral, etc. All these must statelessly display
	only an interpretation of immutable data, so they will be some func of java.awt.Graphics and Funcer,
	which paints the Funcer on the Graphics. None of these uitools will get to access mouse and keyboard events
	anymore, cuz that will have to be done at the Funcer level.
	*/
	
	protected double lastTimeRecorded = 0;
	
	public MouseYRecorder32HzToSerializableFloat1d(){
		addMouseMotionListener(new MouseMotionListener(){
			public void mouseMoved(MouseEvent e){
				double now = Time.now();
				double interval = 1./32;
				if(now+interval > lastTimeRecorded){
					lastTimeRecorded = now; //FIXME this will record a little slower
					mouseFractions[0].add(1-(float)e.getY()/getHeight());
					//mouseFractions[1].add((float)e.getX()/getWidth());
					repaint();
				}
			}
			public void mouseDragged(MouseEvent e){ mouseMoved(e); }
		});
	}
	
	public Serializable dataToSave(){
		//return new float[][]{floatListToArray(mouseFractions[0]), floatListToArray(mouseFractions[0])};
		return floatListToArray(mouseFractions[0]);
	}
	
	public void dataLoad(Serializable data){
		if(!(data instanceof float[])) throw new Error("Not a float[]: "+data);
		mouseFractions = new List[]{ floatArrayToList((float[])data) };
		/*if(!(data instanceof float[][])) throw new Error("Not a float[][]: "+data);
		float[][] d = (float[][]) data;
		if(d.length != 2) throw new Error("Not [2]: "+data);
		if(d[0].length != d[1].length) throw new Error("Diff size float[]s: "+data);
		for(int dim=0; dim<mouseFractions.length; dim++){
			mouseFractions[dim] = floatArrayToList(d[dim]);
		}*/
	}
	
	static List<Float> floatArrayToList(float[] a){
		List<Float> list = new ArrayList();
		for(int i=0; i<a.length; i++) list.add(a[i]);
		return list;
	}
	
	static float[] floatListToArray(List<Float> list){
		float[] a = new float[list.size()];
		for(int i=0; i<a.length; i++) a[i] = list.get(i);
		return a;
	}
	
	public void paint(Graphics g){
		int w = getWidth(), h = getHeight();
		double maxRadius = Math.min(w, h)/2;
		double centerX = w/2, centerY = h/2;
		g.setColor(Color.black);
		g.fillRect(0, 0, w, h);
		int turns = 5;
		int samples = mouseFractions[0].size();
		for(int dim=0; dim<mouseFractions.length; dim++){
			int[] y = new int[samples];
			//int[] yB = new int[samples]; //TODO display mouseX input. For now, just recording it.
			int[] x = new int[samples];
			//int[] xB = new int[samples];
			for(int s=0; s<samples; s++){
				double timeFraction = (double)s/(samples-1);
				double angle = 2*Math.PI*turns*timeFraction;
				float sample = mouseFractions[dim].get(s);
				double radius = .2+.6*maxRadius*timeFraction+sample*20; //FIXME this is half a turns radius turn too far on both ends (so put the .2+.6* but theres an exact way to do it.
				y[s] = (int)(centerY + radius*-Math.cos(angle));
				x[s] = (int)(centerX + radius*Math.sin(angle));
			}
			g.setColor(dim==0 ? Color.green : new Color(.5f, .5f, 1f));
			for(int s=1; s<samples; s++){
				g.drawLine(x[s-1], y[s-1], x[s], y[s]);
			}
		}
	}
	
	public void accept(Object o){
		//TODO save and load float[] (raw, not Serializable) or other interactions with rest of system
	}
	
	/*public double spiralCoordinateToY(double time, double amplitude){
		throw new Error("TODO");
	}
	
	public double spiralCoordinateToX(double time, double amplitude){
		throw new Error("TODO");
	}
	
	public double yxToSpiralCoordinateTime(double y, double x){
		throw new Error("TODO");
	}
	
	public double yxToSpiralCoordinateAmplitude(double y, double x){
		throw new Error("TODO");
	}*/
	
	public static void main(String[] args){
		ScreenUtil.testDisplayWithExitOnClose(new MouseYRecorder32HzToSerializableFloat1d());
	}
	
	

}
