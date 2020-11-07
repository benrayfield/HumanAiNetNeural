/** Ben F Rayfield offers this software opensource MIT license */
package start;
import static mutable.util.Lg.*;

import javax.sound.midi.Track;

import immutable.datasets.TimelessExperimentUtil;
//import immutable.occamsfuncer_old.Funcer;
//import immutable.occamsfuncer_old.RootLoop;
//import immutable.occamsfuncer.util.OccamsfuncerParser;
import immutable.util.Text;
import mutable.learnloop_partsMovedToMutableuitoollearnloop.old.ui.PaintSlidingVecUi;
import mutable.listweb.start.StartSingleUserWindow;
import mutable.recurrentjavaui.LstmViewer;
import mutable.util.Files;
import mutable.util.Lg;

public class Start{
	public static void main(String[] args){
		//TestRemoteServerLag.main(args); //FIXME remove this test soon
		//System.exit(0);
		
		
		
		Lg.lg("java.library.path="+System.getProperty("java.library.path"));
		
		//lg("testtvq="+TimelessExperimentUtil.trainingVecQuantity("mnistOcrTestFile16x16From28x28ShrunkToHalfSizeAndOnehotLabelsAddedAlongASizeExpandingTo16x16"));
		
		//FIXME setting java.library.path might fail here cuz of including jars on classpath in eclipse?
		//System.setProperty("java.library.path", Files.libDir.getAbsolutePath());
		
		//TODO rsrch mindmap ===perBatch ===perWeight etc
		//merging some code from PaintSlidingVecUi,
		//in research mindmap.
		//then bring back maxradius norming.
		
		Lg.todo("Use the other occamsfuncer in newer dir in eclipse workspace, not this one");
		Lg.todo(""+LstmViewer.class);
		
		
		StartSingleUserWindow.main(args);
		//PaintSlidingVecUi.main(args);

		//TODO move this to something called from listweb/StartSingleUserWindow while(true){
		//	RootLoop.nextState();
		//}
		
		//TODO design everything around immutable.merkleforest.Merk which is the new core of everything.
		
		
		
		
		//TODO I need occamsfuncer hooked into listwebresearch,
		//as jsoundcard plugin and another way as instant testcase that returns a value to screen.
		

	}
}
