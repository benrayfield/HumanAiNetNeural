package mutable.listweb.ui;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.util.Arrays;

import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.ListCellRenderer;

import mutable.listweb.ListwebRoot;
import mutable.listweb.ListwebUtil;
import mutable.listweb.Options;

public class PrilistRender extends JPanel implements ListCellRenderer{
	
	//TODO merge duplicate code between PrilistRender and StackRender
	
	public final Color textSelected, textNormal, backgroundSelected, background;
	
	public final boolean reverse;
	
	protected final JLabel timeLabel, nameLabel;
	
	public PrilistRender(boolean reverse, Color textSelected, Color textNormal, Color backgroundSelected, Color background){
		super(new BorderLayout());
		this.reverse = reverse;
		this.textSelected = textSelected;
		this.textNormal = textNormal;
		this.backgroundSelected = backgroundSelected;
		this.background = background;
		timeLabel = new JLabel("TODO time goes here");
		nameLabel = new JLabel("TODO name goes here");
		timeLabel.setFont(timeFont);
		nameLabel.setFont(nameFont);
		add(timeLabel,BorderLayout.WEST);
		add(nameLabel, BorderLayout.CENTER);
	}
	
	public static final Font timeFont = new Font(Font.MONOSPACED, 0, 12);
	
	public static final Font nameFont = new Font(Font.DIALOG, Font.BOLD, 12);
	
	/*static final Color bgHdwmy = new Color(0x99ddff);
	static final Color bgHdwm = new Color(0x88ccff);
	static final Color bgHdw = new Color(0x7799ff);
	static final Color bgHd = new Color(0x6677ff);
	static final Color bgH = new Color(0x5555ff);
	static final Color bgNoPriority = new Color(0x4444dd);
	*/
	static final Color bgHdwmy = new Color(0xffffff);
	static final Color bgHdwm = new Color(0xdddddd);
	static final Color bgHdw = new Color(0xcccccc);
	static final Color bgHd = new Color(0xbbbbbb);
	static final Color bgH = new Color(0xaaaaaa);
	static final Color bgNoPriority = new Color(0x999999);
	
	static final Color bgIfIgnoringHdwmy = bgHdwmy;
	
	public Component getListCellRendererComponent(
			JList list, Object value, int index, boolean isSelected, boolean cellHasFocus){
		//TODO merge duplicate code between StackRender and PrilistRender
		String name = (String)value;
		
		//return new JLabel("cellrendering "+name);
		//setText("self cellrendering "+name);
		//return this;
		
		
		
		int selectedIndex = list.getSelectedIndex();
		boolean selected = index==selectedIndex;
		
		Color bg;
		
		boolean optionUseTodotimesAndHdwmyColors =
			Options.option(ListwebUtil.optionUseTodotimesAndHdwmyColors, false);
		if(optionUseTodotimesAndHdwmyColors){
			double todoTime = ListwebRoot.getTodoTime(name);
			timeLabel.setText(TodoTimeEditable.timeToString(todoTime)+' ');
			bg = selected ? backgroundSelected : background;
			if(!selected){
				try{
					String def = ListwebRoot.def(name);
					if(def.startsWith("!hdwmy")) bg = bgHdwmy;
					else if(def.startsWith("!hdwm")) bg = bgHdwm;
					else if(def.startsWith("!hdw")) bg = bgHdw;
					else if(def.startsWith("!hd")) bg = bgHd;
					else if(def.startsWith("!h")) bg = bgH;
					else bg = bgNoPriority;
				}catch(Throwable t){ //ListwebRoot.def(name); throws if name is too long
					//leave as other color.
				}
				//There are a few names so long the program crashes when load them,
				//or at least when load in certain combos that seem not to happen anymore (as of 2018-12-12),
				//but at least this should prevent the crashing thats been happening.
				//I'll fix the long names eventually.
			}
		}else{
			timeLabel.setText(TodoTimeEditable.timeToStringDisabled+' ');
			bg = selected ? backgroundSelected : bgIfIgnoringHdwmy;
		}
		nameLabel.setText(name);
		
		setBackground(bg);
		timeLabel.setBackground(selected ? backgroundSelected : bg);
		nameLabel.setBackground(selected ? backgroundSelected : bg);
		timeLabel.setForeground(selected ? textSelected : textNormal);
		nameLabel.setForeground(selected ? textSelected : textNormal);
		return this;
		
	}

}
