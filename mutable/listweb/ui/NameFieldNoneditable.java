package mutable.listweb.ui;
import java.awt.Dimension;
import java.util.function.Consumer;

import javax.swing.JTextArea;
import javax.swing.JTextField;

import mutable.listweb.ListwebRoot;

/** listens (Consumer<String>) for changes to aSelectPtr.selectedName
and displays it noneditable. This is normally used above a DefPanel
andOr other editors of properties of the selectedName.
*/
public class NameFieldNoneditable extends JTextArea implements Consumer<String>{
	
	public final String selectPtr;
	
	public NameFieldNoneditable(String selectPtr){
		//super(3,0);
		setLineWrap(true);
		setWrapStyleWord(false);
		//setMinimumSize(new Dimension(5,0));
		//setPreferredSize(new Dimension(100,0));
		this.selectPtr = selectPtr;
		setEditable(false);
		ListwebRoot.startListening(this, selectPtr);
	}

	public void accept(String aSelectPtr){
		if(aSelectPtr.equals(selectPtr)){
			String selectedName = (String) ListwebRoot.get(selectPtr).get("selectedName");
			setText(selectedName==null ? "" : selectedName);
			invalidate();
		}
	}
	
	/*public void setMinimumSize(Dimension minimumSize){
		super.setMinimumSize(new Dimension(100, minimumSize.height));
	}
	
	public void setPreferredSize(Dimension preferredSize){
		super.setPreferredSize(new Dimension(100, preferredSize.height));
	}*/

}