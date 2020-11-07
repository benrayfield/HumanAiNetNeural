package mutable.uitool.util;
import static mutable.util.Lg.*;
import java.awt.Dimension;

import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import mutable.uitool.Uitool;

public class DisplayString extends Uitool{
	
	public final JTextArea textarea;
	
	public DisplayString(){
		textarea = new JTextArea();
		textarea.setWrapStyleWord(true);
		textarea.setLineWrap(true);
		textarea.setEditable(false);
		textarea.setPreferredSize(new Dimension(300,300));
		add(new JScrollPane(textarea));
	}
	
	public void accept(Object o){
		lg("DisplayString this.hashcode="+this.hashCode()+" accept(Object)="+o);
		textarea.setText(o==null ? "null" : o.toString());
	}

}
