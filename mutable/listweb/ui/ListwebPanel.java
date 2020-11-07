package mutable.listweb.ui;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Arrays;

import javax.swing.*;
import javax.swing.plaf.SplitPaneUI;

import mutable.listweb.ListwebRoot;
import mutable.listweb.start.StartSingleUserWindow;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.tools.ResearchPanel;

/** A TwoPrilistsPanel and a def editor separated by a splitpane.
Choose any 3 mindmap names (which can exist or not) to be upStack, downStack, and selectPtr,
where state about the ui is stored.
*/
public class ListwebPanel extends JSplitPane{
	
	public final ResearchPanel rp;
	
	public ListwebPanel(ResearchPanel rpOrNull, String upStackName, String downStackName, String selectPtrName, String nameOfSearchResults){
		super(JSplitPane.VERTICAL_SPLIT);
		this.rp = rpOrNull;
		Arrays.asList(upStackName, downStackName, selectPtrName, nameOfSearchResults)
			.stream().forEach(ListwebRoot::vervarDisable);
		if(ListwebRoot.prilist(upStackName).isEmpty()) ListwebRoot.addToEndOfPrilistIfNotExist(upStackName, ListwebRoot.rootName);
		if(ListwebRoot.prilist(downStackName).isEmpty()) ListwebRoot.addToEndOfPrilistIfNotExist(downStackName, ListwebRoot.rootName);
		if(ListwebRoot.get(upStackName).get("selectedName") == null) ListwebRoot.setWithEvent(upStackName, "selectedName", ListwebRoot.rootName);
		if(ListwebRoot.get(downStackName).get("selectedName") == null) ListwebRoot.setWithEvent(downStackName, "selectedName", ListwebRoot.rootName);
		TwoPrilistsPanel p = new TwoPrilistsPanel(upStackName, downStackName, selectPtrName, nameOfSearchResults){
			public Dimension getMinimumSize(){
				return new Dimension(20,20);
			}
			
			public Dimension getPreferredSize(){
				return new Dimension(20,20);
			}
		};
		DefPanel def = new DefPanel(selectPtrName);
		//JSplitPane vsplit = new JSplitPane(JSplitPane.VERTICAL_SPLIT);		
		JPanel lowPanel = new JPanel(){
			public Dimension getMinimumSize(){
				return new Dimension(20,20);
			}
			
			public Dimension getPreferredSize(){
				return new Dimension(20,20);
			}
		};
		lowPanel.setLayout(new BorderLayout());
		//TodoTimeEditable tte = new TodoTimeEditable(selectPtrName);
		NameFieldNoneditable n = new NameFieldNoneditable(selectPtrName);
		//lowPanel.add(new JScrollPane(n, JScrollPane.VERTICAL_SCROLLBAR_NEVER, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER), BorderLayout.NORTH);
		//JPanel aboveDef = new JPanel(new GridLayout(0,1));
		//aboveDef.add(n);
		//aboveDef.add(tte);
		//lowPanel.add(aboveDef, BorderLayout.NORTH);
		JPanel buttonsAndNamePanel = new JPanel(new BorderLayout());
		JPanel buttonsPanel = new JPanel(new GridLayout(1,0));
		
		//TODO save and load using Var of some selectPtr instead of accessing the uis.
		//Make another Var for the evaling stuff on right side of screen.
		//Actually instead do it the way DefPanel does.
		
		JButton buttonLeftLeft = new JButton(new AbstractAction("<<<"){
			public void actionPerformed(ActionEvent e){
				if(rpOrNull != null){
					lg("<<<SAVE...");
					String selectedName = (String) ListwebRoot.get(StartSingleUserWindow.select).get("selectedName");
					String tabNameInResearchPanel = rpOrNull.selectedTabNameOrNull();
					if(tabNameInResearchPanel.equals(ResearchPanel.displayNameOfStartTab)){
						//TODO merge duplicate code
						String data = ListwebRoot.def(StartSingleUserWindow.researchPanelTextContainerStart);
						ListwebRoot.setDef(selectedName, data);
					}else if(tabNameInResearchPanel.equals(ResearchPanel.displayNameOfContinueTab)){
						//TODO merge duplicate code
						String data = ListwebRoot.def(StartSingleUserWindow.researchPanelTextContainerContinue);
						ListwebRoot.setDef(selectedName, data);
						ListwebRoot.setDef(StartSingleUserWindow.researchPanelTextContainerStart, data); //copy from continue textarea to start textarea
					}
				}
			}
		});
		buttonLeftLeft.setToolTipText("Copy text from right to left,saving it, replacing what was in left");
		buttonsPanel.add(buttonLeftLeft);
		JButton buttonRightRight = new JButton(new AbstractAction(">>>"){
			public void actionPerformed(ActionEvent e){
				if(rpOrNull != null){
					lg("LOAD>>>...");
					String selectedName = (String) ListwebRoot.get(StartSingleUserWindow.select).get("selectedName");
					String data = ListwebRoot.def(selectedName);
					//SwingUtilities.invokeLater(()->{
						ListwebRoot.setDef(StartSingleUserWindow.researchPanelTextContainerStart, ""); //so if data is the same text, theres still an event
						ListwebRoot.setDef(StartSingleUserWindow.researchPanelTextContainerStart, data);
						ListwebRoot.setDef(StartSingleUserWindow.researchPanelTextContainerContinue, data);
					//});
				}
			}
		});
		buttonRightRight.setToolTipText("Copy text from left to right, executing it, same as if you typed in right");
		buttonsPanel.add(buttonRightRight);
		/*buttonsPanel.add(new JButton(new AbstractAction("TEST"){
			public void actionPerformed(ActionEvent e){
				String selectedName = (String) ListwebRoot.get(StartSingleUserWindow.select).get("selectedName");
				String data = "testRAND_"+Rand.strongRand.nextLong();
				ListwebRoot.setDef(selectedName, data);
			}
		}));*/
		//TODO also load when middleclick a mindmap name
		
		
		buttonsAndNamePanel.add(buttonsPanel, BorderLayout.NORTH);
		buttonsAndNamePanel.add(n, BorderLayout.CENTER);
		lowPanel.add(buttonsAndNamePanel, BorderLayout.NORTH);
		lowPanel.add(def, BorderLayout.CENTER);
		add(p);
		add(lowPanel);
		//setUI((SplitPaneUI)null);
		setContinuousLayout(true);
		//setLayout(new GridLayout()); //TODO should this just be a JSplitPane?
		//setLayout(new BorderLayout());
		//add(vsplit, BorderLayout.SOUTH);
		setResizeWeight(.8); //FIXME why does this have no effect sometimes?
	}
	
	

}
