package mutable.tools;

import java.awt.GridLayout;

import javax.swing.JPanel;

import mutable.listweb.ListwebRoot;

/** multiple experiments side by side, each named by a mindmapname.
The text after my class name is those mindmapnames separated by whitespace.
*/
public class PanelsHorizontal extends JPanel{
	public PanelsHorizontal(String code){
		super(new GridLayout(1,0));
		String[] tokens = code.trim().split("\\s");
		for(String token : tokens){
			add(Eval.evalToUitool(ListwebRoot.def(token)));
		}
	}
}
