package mutable.tools;
import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.AbstractSpinnerModel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.event.ChangeListener;

/** contains a JTextArea and a numberfield which views and edits any number selected in the textarea.
Also I might put in other tools and ways of editing.
This is designed for use in ResearchPanel.
*/
public class ResearchTextEditor extends JPanel{
	
	public final JTextArea textEditor;
	
	public final JSpinner numberEditor;
	
	protected double scalarMult;
	
	/** scalarMult is how much to multiply or divide a scalar when editing it with the
	numberEditor up/down buttons, if its a scalar, but if its integer its plus/minus one.
	*/
	public ResearchTextEditor(double scalarMult){
		super(new BorderLayout());
		textEditor = new JTextArea();
		numberEditor = new JSpinner(/*new AbstractSpinnerModel(){
			protected double value;
			public void setValue(Object o){
				double newValue = ((Number)o).doubleValue();
				if(value != newValue){
					value = newValue;
					fireStateChanged();
				}
			}
			
			public Object getValue(){
				return value;
			}
			
			public Object getPreviousValue(){
				if((long)value == value) return value-1;
				return value/scalarMult;
			}
			
			public Object getNextValue(){
				if((long)value == value) return value+1;
				return value*scalarMult;
			}
		}*/){
			public Object getNextValue(){
				double value = ((Number)getValue()).doubleValue();
				if((long)value == value) return value-1;
				return value*scalarMult;
		        //return getModel().getNextValue();
		    }
			public Object getPreviousValue(){
				double value = ((Number)getValue()).doubleValue();
				if((long)value == value) return value-1;
				return value/scalarMult;
		        //return getModel().getPreviousValue();
		    }
		};
		Dimension d = new Dimension(70,20);
		numberEditor.setMinimumSize(d);
		numberEditor.setPreferredSize(d);
		add(textEditor, BorderLayout.CENTER);
		add(numberEditor, BorderLayout.EAST);
		numberEditor.setEnabled(false); //FIXME numberEditor doesnt work the way I want, and isnt hooked into textEditor
		//TODO connect textEditor and numberEditor
	}
}
