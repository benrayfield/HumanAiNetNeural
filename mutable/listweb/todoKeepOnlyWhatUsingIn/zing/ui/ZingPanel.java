package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ui;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;

import mutable.listweb.Debug;
import mutable.listweb.ListwebRoot;
import mutable.listweb.ListwebUtil;
import mutable.listweb.start.StartSingleUserWindow;
import mutable.util.ui.ScreenUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.ui.DefPanel;
import mutable.listweb.ui.ListwebPanel;
import mutable.listweb.ui.NameFieldNoneditable;
import mutable.listweb.ui.TwoPrilistsPanel;

/** Has up to 1 of each kind of io and ui and (TODO) expandable with new kinds at runtime.
The main 4 will be listweb (TODO integrate with zing instead of separate system),
ZingVideo, JSoundCard (TODO), and vector (TODO).
For now, its a hardcoded listweb and ZingVideo.
<br><br>
Starts with no ZingScreen but it (and thread of repaint events)
can be toggled using enableScreen(boolean).
*/
public class ZingPanel extends JPanel{
	
	protected Thread thread;
	
	protected final ListwebPanel listwebPanel;
	
	protected final ZingVideoAftrans screen;
	
	public ZingPanel(String upStackName, String downStackName, String selectPtrName, String nameOfSearchResults){
		setLayout(new BorderLayout());
		//JSplitPane horiz super(JSplitPane.HORIZONTAL_SPLIT, true);
		//JSplitPane horiz = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, true);
		listwebPanel = new ListwebPanel(null, upStackName, downStackName, selectPtrName, nameOfSearchResults);
		//JTabbedPane tabs = new JTabbedPane();
		//tabs.add(new ZingVideo(), "pixels");
		//horiz.add(tabs);
		screen = new ZingVideoAftrans();
		ZingVideoAftrans.testDisplay(screen);
		enableScreen(false);
		//add(lw);
		//add(z);
		//setResizeWeight(.3);
		//setResizeWeight(.5);
		//horiz.setOneTouchExpandable(true);
		//setLayout(new GridLayout());
		//add(horiz);
	}
	
	/** the ZingVideo per-pixel screen area on the right. Displays it and schedules repaints, or not.
	If repaints are scheduled, they still become very slow to save compute resources
	after not moving mouse over it for long enough.
	*/
	public synchronized void enableScreen(boolean enable){
		removeAll();
		if(enable){
			JSplitPane horiz = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, true);
			horiz.add(listwebPanel);
			horiz.add(screen);
			add(horiz, BorderLayout.CENTER);
			horiz.setResizeWeight(.3);
			thread = new Thread(){
				public void run(){
					while(!isInterrupted()){
						//lg("listweb minsize: "+lw.getMinimumSize());
						lg("\r\n\r\n");
						displayMinSizesRecursive("", listwebPanel);
						Time.sleepNoThrow(3);
					}
				}
			};
			thread.start();
		}else{
			if(thread != null) thread.interrupt();
			thread = null;
			add(listwebPanel, BorderLayout.CENTER);
		}
	}
	
	static void displayMinSizesRecursive(String lgPrefix, Component j){
		if(500 < j.getMinimumSize().getWidth()){
			lg(lgPrefix+"component "+j.getClass().getName()+" sizes");
			lg(lgPrefix+"min="+j.getMinimumSize());
			lg(lgPrefix+"prf="+j.getPreferredSize());
			lg(lgPrefix+"max="+j.getMaximumSize());
			if(j instanceof Container){
				Container jj = (Container)j;
				for(Component child : jj.getComponents()){
					displayMinSizesRecursive(lgPrefix+'\t', child);
				}	
			}
		}
	}

}