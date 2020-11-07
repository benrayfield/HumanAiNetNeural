package mutable.listweb.ui;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.ListCellRenderer;

import mutable.listweb.ListwebRoot;

public class StackRender implements ListCellRenderer{
	
	//TODO merge duplicate code between PrilistRender and StackRender
	
	public final Color background, textActive, textSelected, textGhost;
	
	public final boolean reverse;
	
	public StackRender(boolean reverse, Color background, Color textActive, Color textSelected, Color textGhost){
		this.reverse = reverse;
		this.background = background;
		this.textActive = textActive;
		this.textSelected = textSelected;
		this.textGhost = textGhost;
	}
	
	public Component getListCellRendererComponent(
			JList list, Object value, int index, boolean isSelected, boolean cellHasFocus){
		//TODO merge duplicate code between StackRender and PrilistRender
		String name = (String)value;
		int selectedIndex = list.getSelectedIndex();
		boolean selected = index==selectedIndex;
		boolean ghost = selectedIndex==-1 || (reverse ? index<selectedIndex : index>selectedIndex);
		double todoTime = ListwebRoot.getTodoTime(name);
		JLabel timeLabel = new JLabel(TodoTimeEditable.timeToString(todoTime)+' ');
		JLabel nameLabel = new JLabel(name);
		
		Color bg = background;
		//FIXME why isnt this affecting background color in stack on screen?
		//if(!selected){
			String def = ListwebRoot.def(name);
			if(def.startsWith("!hdwmy")) bg = PrilistRender.bgHdwmy;
			else if(def.startsWith("!hdwm")) bg = PrilistRender.bgHdwm;
			else if(def.startsWith("!hdw")) bg = PrilistRender.bgHdw;
			else if(def.startsWith("!hd")) bg = PrilistRender.bgHd;
			else if(def.startsWith("!h")) bg = PrilistRender.bgH;
			else bg = PrilistRender.bgNoPriority;
		//}
		
		//nameLabel.setBackground(background);
		nameLabel.setBackground(bg);
		nameLabel.setForeground(selected ? textSelected : (ghost ? textGhost : textActive));
		JPanel panel = new JPanel(new BorderLayout());
		panel.add(timeLabel,BorderLayout.WEST);
		panel.add(nameLabel, BorderLayout.CENTER);
		timeLabel.setFont(PrilistRender.timeFont);
		nameLabel.setFont(PrilistRender.nameFont);
		return panel;
	}

}
