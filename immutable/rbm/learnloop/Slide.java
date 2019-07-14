package immutable.rbm.learnloop;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NavigableMap;
import java.util.Random;
import java.util.function.UnaryOperator;

import immutable.rbm.learnloop.slideactions.AddLearningVec;

/** Immutable. Approx a timewindow of trainingVectors that are added on one side and are removed
when they're learned enough, so those removed tend to be near the other end of the slide.
This has a RBM learning them and stats on that. This is a gpu-optimized process (lwjgl opencl)
which does prediction in CPU for low lag (about 50 hz) and learning in GPU (about 3 hz).
*/
public class Slide{
	
	public final RBM rbm;
	
	/** immutable List of immutable LearningVecs */
	public final List<LearningVec> lvecs;
	
	/** immutable List of immutable SlideActions (such as add LearningVec) after GPU learning cycle ends.
	Merged this here from SlideSync class whose comment was...
	QUOTE Immutable. Whatever changes are to be done to a Slide,
	queue them here, and when GPU code finishes its next learning cycle, merge them.
	Example: adding LearningVecs. The wrapper of GPU learning code depends on the List<LearningVec> indexs.
	That code will be modified to synchronize on SlideSync which is fast to forkEdit.
	Also its important to not make many small forkEdits to a list of potentially thousands of LearningVecs,
	which is not a bottleneck when learning in batches. UNQUOTE.
	*/
	public final List<SlideAction> sync;
	
	/** Other parts than go in cuzstom javaclasses (such as List<SlideAction>, List<LearningVec>, and RBM).
	This is an OccamsJsonDS object. Eventually more (or all?) things will be occamsfuncer instead.
	<br><br>
	"selectedLvecIndex" key is (double) 0 to lvecs.size()-1.
	That changes when mouseover any of the
	<br><br>
	"displayModes: key:
	Display all these modes at once, drawn in reverse of this order in case of overlap.
	Each is a NavigableMap (as in OccamsJsonDS) describing some kind of thing to display.
	Example: {"displayType":"1dScalarWavesHorizontallyOfPredictedAndObserved"}
	Example: {"displayType":"logMapInBottomLeft"} //TODO eval these using beanshell and put java code in the map? Do I want turingcomplete data before sandbox is working?
	Example: {"displayType":"evalCodeOnSlideAndDisplayString", "code":"return \"\"+p;"}
	Example: {"displayType":"rbmNodes"}
	Example: {"displayType":"predicZigzag", "magnify":7, "pixeslOffset":0}
	Example: {"displayType":"learnZigzag", "magnify":7, "pixeslOffset":100}
	Example: {"displayType":"visibleNodesObservedVsCorrectAsRectangleInTopRight", "visibleNodesWide":20, "magnify":5}
	Example: {"displayType":"observedVisibleNodesTiledAsRectangles", "visibleNodesWide":20, "magnify":5}
	//Some of those might change as I figure out how they fit together,
	//like learnZigzag.pixelsOffset would normally depend on predictZigzag number of zigzags, magnify, etc.
	*/
	public NavigableMap other;
	
	//TODO public final UnaryOperator<float[]> confuser; but where to store that? Should it be here?
	
	public Slide(RBM rbm, List<LearningVec> lvecs){
		this(rbm,lvecs,Collections.EMPTY_LIST);
	}
	
	public Slide(RBM rbm, List<LearningVec> lvecs, List<SlideAction> sync){
		this.rbm = rbm;
		this.lvecs = lvecs;
		this.sync = sync;
	}
	
	/** The SlideAction happens when sync(). SlideActions must be fast, to read and write things, not numbercrunching */
	public Slide queue(SlideAction a){
		List<SlideAction> list = new ArrayList(sync.size()+1);
		list.addAll(sync);
		list.add(a);
		return new Slide(rbm,lvecs,Collections.unmodifiableList(list));
	}
	
	/** UPDATE: Generalized to List of SlideAction, such as the action of adding a LearningVec,
	or the action of removing a certain LearningVec or removing those which have learned enough
	or changing some param of the Slide such as the batch size,
	and even the GPU will use the sync system as SET RBM will be an action,
	so this sync func now takes no params.
	...
	Old comment...
	Moves LearningVecs from sync list into main list, and sets RBM. This is called at end of each GPU learning cycle.
	During that cycle, the RBM can be used for prediction (normally 32 times per second)
	and the Slide can be forkEdited such as in a Var<Slide> to add LearningVecs,
	and at the end of GPU learning cycle it will read that Var<Slide>
	and write its RBM (keeping the new LearningVecs) by calling this. That way, they dont wait on eachother.
	...
	OLD...
	public double applyAsDouble(double mouseYVelocity){
		Slide slide = varSlide.get();
		
		
		/*TODO do I want rbm to be be inside a mutable SlidingVecs (a new class refactored out of Slidinglearnrandvecui)?
		No, I want that new class to be immutable, and to have a Var of that here.
		Since learning is done in batches, it being immutable isnt a bottleneck. Create an immutable version of LearningVec.
		Since each cycle of mousepuz will at least do prediction in RBM once,
		any state in mousepuz probably wont be the bottleneck to copy,
		and I am trying to keep it simple. Maybe acyclicFlow, metarnnAcyclicFlow,
		or javassist on sandboxed code? Or ufnode? Ufnode later, its not nearly finished enough,
		and I need to prototype this soon.
		*
		
		TODO
		
		varSlide.set(slide);
		
		TODO how to sync the predictions which occur more often and in a different thread than gpu learning but both write varSlide?
	}
	*/
	public Slide sync(){
		List<LearningVec> addVecs = new ArrayList();
		for(SlideAction a : sync){
			if(a instanceof AddLearningVec){
				addVecs.add(((AddLearningVec)a).lv);
			}else{
				throw new Error("Unknown type "+a.getClass().getName());
			}
		}
		List<LearningVec> list = new ArrayList(lvecs.size()+sync.size());
		list.addAll(lvecs);
		list.addAll(addVecs);
		return new Slide(rbm, list, Collections.EMPTY_LIST);
	}
	
	/*public Slide nextState(Random rand){	
	}*/
	

}
