package mutable.tools;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.lg;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.util.Arrays;
import java.util.function.Consumer;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.ScrollPaneConstants;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import mutable.listweb.Debug;
import mutable.listweb.ListwebRoot;
import mutable.listweb.start.StartSingleUserWindow;
import mutable.uitool.Uitool;
import mutable.uitool.util.DisplayString;

/** has a textarea that you type (or automatically listweb puts there) text
that starts with uitool: then a class name then the rest of the string
goes into constructor of that class and if its a java.awt.Component
then displays it. Thats done by ListwebAIResearchTool.eval(String)
which would return (for example) a Component,
but it doesnt have to be a Component, though this class might not know
what to do with it if its not.
<br><br>
I'm upgrading this class to be similar to DefEditor in its events.
*/
public class ResearchPanel extends JPanel implements Consumer<String>{
	
	public final String textContainerStart, textContainerContinue;
	
	//public final JTextArea textStart, textContinue;
	public final ResearchTextEditor textStart, textContinue;
	
	
	protected boolean listenToDocument = true;
	
	//protected final JScrollPane scroll;
	
	protected JTabbedPane tabs;
	
	public String selectedTabNameOrNull(){
		int i = tabs.getSelectedIndex();
		if(i == -1) return null;
		return tabs.getTitleAt(i);
	}
	
	public static final String displayNameOfStartTab = "Start", displayNameOfContinueTab = "Continue";
	
	/** save and load state in selectPtr in ListwebRoot */
	public ResearchPanel(String textContainerStart, String textContainerContinue){
		super(new BorderLayout());
		this.textContainerStart = textContainerStart;
		this.textContainerContinue = textContainerContinue;
		//textStart = new JTextArea();
		//textContinue = new JTextArea();
		double scalarMult = 1.1;
		textStart = new ResearchTextEditor(scalarMult);
		textContinue = new ResearchTextEditor(scalarMult);
		//add(text, BorderLayout.NORTH);
		/*text.getDocument().addDocumentListener(new DocumentListener(){
			public void removeUpdate(DocumentEvent e){
				//lg("remove");
				onTextChanged();
			}
			public void insertUpdate(DocumentEvent e){
				//lg("insert");
				onTextChanged();
			}
			public void changedUpdate(DocumentEvent e){
				//lg("changed");
				onTextChanged();
			}
		});
		*/
		//add(tfName, BorderLayout.NORTH);
		//tfName.setEditable(false);
		//int v = ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS;
		//int h = ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER;
		//add(scroll = new JScrollPane(text, v, h), BorderLayout.NORTH);
		tabs = new JTabbedPane();
		tabs.add(displayNameOfStartTab, textStart);
		tabs.add(displayNameOfContinueTab, textContinue);
		tabs.setSelectedIndex(0);
		add(tabs, BorderLayout.NORTH);
		for(ResearchTextEditor editor : new ResearchTextEditor[]{textStart, textContinue}){
			//text.setText(Eval.exampleToEval);
			editor.textEditor.getDocument().addDocumentListener(new DocumentListener(){
				public void insertUpdate(DocumentEvent e){
					onTextChanged(ResearchPanel.this, editor);
				}
				public void removeUpdate(DocumentEvent e){
					onTextChanged(ResearchPanel.this, editor);
				}
				public void changedUpdate(DocumentEvent e){}
			});
			editor.setMinimumSize(new Dimension(50,50));
			editor.textEditor.setLineWrap(true);
			editor.textEditor.setWrapStyleWord(true);
			editor.textEditor.setEditable(true);
		}
		
		//instead of startListening cuz the event causes second Eval.%(String)
		ListwebRoot.startListeningWithoutInstantEvent(this, textContainerStart);
		ListwebRoot.startListeningWithoutInstantEvent(this, textContainerContinue);
	}
	
	protected void finalize() throws Throwable{
		ListwebRoot.stopListening(this, textContainerStart);
		ListwebRoot.stopListening(this, textContainerContinue);
	}
	
	protected void onTextChanged(ResearchPanel rp, ResearchTextEditor editor){
		String t = editor.textEditor.getText();
		try{
			if(editor == textStart){
				//if(!t.trim().endsWith(";")) return; //ignore until put ; at end of textarea
				Object evaled = Eval.$(t); //calls Uitool.accept(t) after creating Uitool
				Component c = asUitool(evaled);
				if(getComponentCount() > 1){
					Uitool u = (Uitool) getComponent(1);
					lg("Closing Uitool "+u);
					u.close(); //in case its doing something in a timed loop
					remove(1);
				}
				add(c,BorderLayout.CENTER);
			}else if(editor == textContinue){
				String theStartText = textStart.textEditor.getText();
				if(!theStartText.equals("") && !t.equals("") && !t.equals(theStartText)){
					//Avoiding this when theStartText is "" is cuz thats used to clear the last Uitool
					//and should not have Continue event. Similar for !t.equals("").
					//!t.equals(theStartText) is for...
					//Dont continue right after >>> button copies the text into Start tab and Continue tab,
					//which for example java:5+6 in Start creates DisplayString of "11"
					//but without this IF, Continue would then DisplayString.accept("java:5+6").
					//Continue tab always does Uitool.accept(what user typed)
					//and is not useful for DisplayString like its useful for telling a neuralnet
					//experiment to update its params during learning on screen.
					if(getComponentCount() > 1){
						Uitool u = (Uitool) getComponent(1);
						u.accept(t);
					}
				}
			}else{
				throw new Error("editor="+editor);
			}
			editor.textEditor.setBackground(Color.white);
			rp.validate();
			rp.repaint();
		}catch(Throwable th){
			editor.textEditor.setBackground(new Color(.8f, .8f, .8f));
			if(th.getMessage() != null && th.getMessage().startsWith("Parse error at line")){
				lg(th.getMessage());
			}else{
				th.printStackTrace(System.err);
			}
		}
		//if(editor == textStart){
			//FIXME should this be for either editor? The other only sends a message
			//to the existing Uitool, doesnt replace the Uitool.
			if(listenToDocument){
				String newDef = editor.textEditor.getText();
				if(editor == textStart){
					lg("ResearchPanel setDef "+textContainerStart+" def["+newDef+"]");
					ListwebRoot.setDef(textContainerStart, newDef);
				}else if(editor == textContinue){
					lg("ResearchPanel setDef "+textContainerContinue+" def["+newDef+"]");
					ListwebRoot.setDef(textContainerContinue, newDef);
				}
			}
		//}
		if(editor == textStart){
			//copy to Continue (FIXME verify it doesnt run the code again)
			ListwebRoot.setDef(StartSingleUserWindow.researchPanelTextContainerContinue, t);
		}
	}
	
	public static Uitool asUitool(Object o){
		//if(o instanceof Component) return (Component) o;
		if(o instanceof Uitool){
			return (Uitool) o;
		}else{// if(o instanceof String){
			Uitool u = new DisplayString();
			u.accept(o);
			return u;
		}
		//throw new Error("Unknown class to asUitool: "+o.getClass().getName());
		//return new JLabel(""+o); //FIXME what if its very long?
		//JPanel p = new JPanel();
		//JLabel j = new JLabel(""+o); //FIXME what if its very long?
		//p.setBackground(Color.green);
		//p.add(j);
		//return p;
	}
	
	public void paint(Graphics g){
		g.setColor(Color.gray);
		g.fillRect(0, 0, getWidth(), getHeight());
		super.paint(g);
	}
	
	public void accept(String aSelectPtr){
		//if(aSelectPtr.equals(textContainer)){
			listenToDocument = false;
			try{
				if(Debug.logSwingLock) lg("TREELOCK researchPanelTextarea");
				synchronized(getTreeLock()){
					//String selectedName = (String) ListwebRoot.get(textContainer).get("selectedName");
					//boolean diffName = selectedPtrSelectedName==null || !selectedPtrSelectedName.equals(selectedName);
					//selectedPtrSelectedName = selectedName;
					//if(selectedName == null){
					//	//tfName.setText("");
					//	text.setText("");
					//	text.setEnabled(false);
					//}else{
					//	text.setEnabled(true);
						//tfName.setText(selectedName);
						ResearchTextEditor editor = null;
						String selectedName = null;
						if(aSelectPtr.equals(textContainerStart)){
							editor = textStart;
							selectedName = StartSingleUserWindow.researchPanelTextContainerStart;
						}
						if(aSelectPtr.equals(textContainerContinue)){
							editor = textContinue;
							selectedName = StartSingleUserWindow.researchPanelTextContainerContinue;
						}
						String oldUiText = editor.textEditor.getText();
						String newDef = ListwebRoot.def(selectedName);
						if(!oldUiText.equals(newDef)){
							editor.textEditor.setText(newDef);
							//onTextChanged();
							/*if(diffName){ //todo diff var name. its not a "name". copied that code from DefPanel where it was name.
								SwingUtilities.invokeLater(()->{
									scroll.getVerticalScrollBar().setValue(0);
								});
							}*/
						}
				}
			}finally{
				listenToDocument = true;
				if(Debug.logSwingLock) lg("TREEUNLOCK researchPanelTextarea");
			}
		//}
	}

}
