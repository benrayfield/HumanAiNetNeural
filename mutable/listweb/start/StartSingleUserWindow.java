package mutable.listweb.start;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

import javax.imageio.ImageIO;
import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JSplitPane;
import javax.swing.WindowConstants;

import mutable.listweb.*;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.ScreenUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.ui.DefPanel;
import mutable.listweb.ui.ListwebPanel;
import mutable.listweb.ui.NameFieldNoneditable;
import mutable.listweb.ui.TwoPrilistsPanel;
//import mutable.taskplayerui.ExampleTaskUi;
//import mutable.taskplayerui.TaskPlayerUi;
import mutable.tools.ResearchPanel;

public class StartSingleUserWindow{
	
	public static final String up = "defaultUpStack",
		down = "delfaultDownStack",
		select = "defaultSelectPtr", //mindmap name containing the selection pointer
		researchPanelTextContainerStart = "researchPanelTextContainerStart", //mindmap name the right panel's start tab's text goes in
		researchPanelTextContainerContinue = "researchPanelTextContainerContinue", //mindmap name the right panel's continue tab's text goes in
		nameOfSearchResults = "searchResults";
	
	public static void main(String[] args){
		lg(StartSingleUserWindow.class.getName());
		ListwebRoot.boot();
		final JFrame window = new JFrame(ListwebUtil.progName);
		Arrays.asList(up, down, select, researchPanelTextContainerStart, researchPanelTextContainerContinue, nameOfSearchResults)
			.stream().forEach(ListwebRoot::vervarDisable);
		
		//Whatever was running in right column when stopped last time,
		//dont start it again automatically, in case it doesnt end well.
		//You can click the >>> button to start it again.
		ListwebRoot.setDef(researchPanelTextContainerStart,"");
		ListwebRoot.setDef(researchPanelTextContainerContinue,"");
		
		final boolean screenEnabled[] = {false};
		//TaskPlayerUi tpu = new TaskPlayerUi(ExampleTaskUi.class);
		
		
		
		
		//TODO use Var here so can sync between defpanel and researchpanel
		//using researchSelectPtr? Make sure not to save state of researchSelectPtr.
				
				
				
		ResearchPanel rp = new ResearchPanel(researchPanelTextContainerStart, researchPanelTextContainerContinue);
		Runnable toggleScreen = ()->{
			screenEnabled[0] = !screenEnabled[0];
			if(screenEnabled[0]){
				lg(window.getComponent(0));
			}
		};
		//window.setJMenuBar(ListwebUtil.newListwebMenuBar(up, down, null));
		window.setJMenuBar(ListwebUtil.newListwebMenuBar(select, up, down, toggleScreen));
		
		JSplitPane p = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, true);
		p.setDividerLocation(400);
		p.add(new ListwebPanel(rp, up, down, select, nameOfSearchResults));
		//p.add(tpu);
		p.add(rp);
		
		window.add(p);
		
		
		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		//window.setSize(screen.width/4, screen.height-150);
		//window.setSize(screen.width/4, screen.height-50);
		window.setSize(1000,700);
		ScreenUtil.moveToScreenCenter(window);
		window.setLocation((screen.width-window.getWidth())/2, 10);
		//window.setLocation(30,70);
		//window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		window.addWindowListener(new WindowListener(){
			public void windowOpened(WindowEvent e){}
			public void windowClosing(WindowEvent e){
				try{
					lg("Window is closing...");
					ListwebRoot.onClosingProg();
				}finally{
					System.exit(0);
				}
			}
			public void windowClosed(WindowEvent e){}
			public void windowIconified(WindowEvent e){}
			public void windowDeiconified(WindowEvent e){}
			public void windowActivated(WindowEvent e){}
			public void windowDeactivated(WindowEvent e){}
		});
		byte iconBytes[] = mutable.util.Files.readFileOrInternalRel("/data/humanainet/icon.jpg");
		try{
			window.setIconImage(ImageIO.read(new ByteArrayInputStream(iconBytes)));
		}catch(IOException e){ throw new Err(e); }
		window.setVisible(true);
	}

}
