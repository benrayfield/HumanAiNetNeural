package mutable.listweb;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.*;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.util.ui.Ask;

public class ListwebUtil{
	private ListwebUtil(){}
	
	public static final String progName = "Human AI Net";
	public static final String licenseName = "MIT";

	public static String unescapeName(String escaped){
		return unescape(escaped,true);
	}
	
	public static String unescapeUrl(String escaped){
		return unescape(escaped,false);
	}
	
	public static String unescape(String escaped, boolean dropUnderscores){
		byte b[] = Text.stringToBytes(escaped);
		//each escape shortens by 2. Each _ is removed, but you can include _ as %5f
		byte b2[] = new byte[b.length];
		int b2Siz = 0;
		for(int i=0; i<b.length; i++){
			if(i<b.length-2 && b[i] == '%' && Text.isHexDigit(b[i+1]) && Text.isHexDigit(b[i+2])){
				b2[b2Siz++] = (byte)(Text.hexDigitToInt(b[++i])<<4 | Text.hexDigitToInt(b[++i]));
			}else if(!dropUnderscores || b[i] != '_'){
				b2[b2Siz++] = b[i];
			}
		}
		byte b3[] = new byte[b2Siz];
		System.arraycopy(b2, 0, b3, 0, b2Siz);
		return Text.bytesToString(b3);
		
		/*try{
			escaped = escaped.replace("_",""); //remove what precedes capitals.
			return URLDecoder.decode(escaped, "UTF-8"); //ERROR changes "+" to " "
		}catch (UnsupportedEncodingException e){
			throw new RuntimeException(e);
		}*/
	}
	
	/** All these are replaced urlEscaping (but on these chars), such as "_" becomes "%5f".
	Also, capitals must be preceded by "_" since names are caseSensitive
	but must also work in fileSystems that check filename equality caseInsensitive (Windows).
	*/
	public static final String escapedChars = "%\\/:;\r\n?*\"<>|._";
	
	/** 2017-9-15 Some of benrayfields existing data is too long when names are escaped. Will fix them later. */
	public static boolean isValidName(String name){
		//capital letters must be prefixed by _ so caseInsensitive file systems can store case sensitive listweb names
		return name.length()!=0 && (name.length() <= ListwebRoot.maxEscapedNameLen/2
			|| escapeNameNoThrow(name).length() <= ListwebRoot.maxEscapedNameLen);
	}
	
	public static String escapeName(String name){
		String escaped = escapeNameNoThrow(name);
		//TODO does any relevant filesystem measure in bytes instead of chars? What max len per path part?
		if(ListwebRoot.maxEscapedNameLen < escaped.length()){
			throw new Err("EscapedName too long: escaped="+escaped+" escapedLen="+escaped.length()+" name="+name);
		}
		return escaped;
	}
	
	/** escapeName is the normal way. This is used when choosing how to rename when its too long. */
	public static String escapeNameNoThrow(String name){
		if(!name.equals(name.trim())){
			throw new Err("Name has leading or trailing whitespace["+name+"]");
		}
		//TODO optimize
		for(int i=0; i<escapedChars.length(); i++){
			name = name.replace(escapedChars.substring(i,i+1),escapeFor(escapedChars.charAt(i)));
		}
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<name.length(); i++){
			char c = name.charAt(i);
			if(Character.isUpperCase(c)) sb.append('_');
			sb.append(c);
		}
		String escaped = sb.toString();
		return escaped;
	}
	
	/** This is used when ready to shorten a name if its too long,
	such as when importing data and copying the full name into def if shortened.
	TODO optimize by adding 2 len for each escapedChars found.
	*/
	public static int charsLenOfEscapedName(String name){
		return escapeNameNoThrow(name).length();
	}
	
	public static int byteLenOfEscapedName(String name){
		return Text.stringToBytes(name).length;
	}

	static String escapeFor(char c){
		String s = Integer.toHexString(c);
		if(s.length() > 2) throw new RuntimeException("TODO urlescape multibyte char: "+c);
		if(s.length() == 1) s = "0"+s;
		return "%"+s;
	}

	/** True if both are null, false if 1 is null, else x.equals(y) */
	public static boolean equals(Object x, Object y){
		return x==null ? y==null : x.equals(y);
	}

	/** If it contains tags, they must be as plain text */
	public static String escapeToAppearAsPlainTextInHtml(String s){
		return s.replace("&","&amp;").replace("<","&gt;").replace(">","&lt;"); //FIXME what are the others?
	}
	
	public static String jsonToSingleLine(String json){
		return json.replaceAll("(\\r|\\n|\\r\\n)\\t*", "");
	}
	
	public static void main(String[] args){
		String s = "abc"+escapedChars+"def";
		lg("testName:"+s);
		String escaped = escapeName(s);
		lg("escaped: "+escaped);
		String unescaped = unescapeName(escaped);
		lg("unescaped: "+unescaped);
		if(!s.equals(unescaped)) throw new Err("Escape or unescape broken");
	}
	
	/** toggleScreen opens and closes the bigger per-pixel part of the window called "screen" */
	public static JMenuBar newListwebMenuBar(String selectionPtr, String upStackName, String downStackName, Runnable toggleScreen){
		JMenuBar menu = new JMenuBar();
		JMenu actions = new JMenu("Actions");
		actions.add(new JMenuItem(
			new AbstractAction("Include new files added externally, and add all files to their first letter/symbol like theFile goes in t"){
				public void actionPerformed(ActionEvent e){
					ListwebRoot.updateRootChars();
				}
			}
		));
		actions.add(new JMenuItem(
			new AbstractAction("Maintenance make sure if x is in y's list, y is in x's list (automatic but slow)"){
				public void actionPerformed(ActionEvent e){
					Maintenance.loadAndSaveAllVarFilesToMakeExistenceInPrilistSymmetric();
				}
			}
		));
		actions.add(new JMenuItem(
			new AbstractAction("Empty defaultUpStack and defaultDownStack (Do this if the 2 middle lists get too big to see outer lists)"){
				public void actionPerformed(ActionEvent e){
					ListwebRoot.clearStack(upStackName, ListwebRoot.rootName);
					ListwebRoot.clearStack(downStackName, ListwebRoot.rootName);
				}
			}
		));
		actions.add(new JMenuItem(
			new AbstractAction("Reduce !hdwmy to !hdwm, and replace !hd to !h, etc."){
				public void actionPerformed(ActionEvent e){
					ListwebRoot.reduceHdwmy();
				}
			}
		));
		actions.add(new JMenuItem(
			new AbstractAction("permDeleteIncludingMyHistoryButNotHistoryOfMeBeingInOthers"){
				public void actionPerformed(ActionEvent e){
					String selected = (String) ListwebRoot.get(selectionPtr).get("selectedName");
					if(Ask.yesNo("permDeleteIncludingMyHistoryButNotHistoryOfMeBeingInOthers "+selected+"?")){
						ListwebRoot.permDeleteIncludingMyHistoryButNotHistoryOfMeBeingInOthers(selected);
					}
				}
			}
		));
		String putTodoTimesMenuName = "Put all names with todoTimes in a new list then clear all todoTimes (for if it gets so big its not worth using)";
		actions.add(new JMenuItem(
			new AbstractAction("Put all names with todoTimes in a new list then clear all todoTimes (for if it gets so big its not worth using)"){
				public void actionPerformed(ActionEvent e){
					String newName = JOptionPane.showInputDialog("Choose name of new list for everything that had a todoTime (then will take some time)");
					if(ListwebUtil.isValidName(newName)){
						List<String> namesWithTodoTimes = ListwebRoot.allNamesWithTodoTimesSortedAscending();
						ListwebRoot.setPrilist(newName, namesWithTodoTimes);
						ListwebRoot.setDef(newName,
							"This was result of the ["+putTodoTimesMenuName+"] menu command at time "+Time.timeStr()+"\r\n\r\n"+ListwebRoot.def(newName));
						ListwebRoot.putAtTopOfPrilist(ListwebRoot.rootName, newName);
						for(String name : namesWithTodoTimes){
							ListwebRoot.setTodoTime(name, 0); //comment of that func says time 0 removes it
						}
					}
				}
			}
		));
		menu.add(actions);
		JMenu options = new JMenu("Options");
		Consumer<String> addBooleanOption = (name)->{
			JCheckBox chk = new JCheckBox(name, Options.option(name,false));
			chk.addActionListener(new ActionListener(){
				public void actionPerformed(ActionEvent e){
					Options.setOption(name, chk.isSelected());
				}
			});
			ListwebRoot.startListening(
				(String listenToName)->{
					boolean oldVal = chk.isSelected();
					boolean newVal = Options.option(name,false);
					if(oldVal != newVal) chk.setSelected(newVal);
				},
				ListwebRoot.rootName
			);
			options.add(chk);
		};
		addBooleanOption.accept("Search in defs (the text lowest in the window for each name), may be slow");
		addBooleanOption.accept("Allow mindmap to run code typed into defs, useful for programmers (SECURITY WARNING)");
		menu.add(options);
		JMenu help = new JMenu("Help");
		help.add(new JLabel("This is a web. Anything can connect to anything."));
		help.add(new JLabel("If x is in y's list, y is in x's list,"));
		help.add(new JLabel("but they can be different orders (try by priority or time)."));
		help.addSeparator();
		help.add(new JLabel("Left click goes into a name"));
		help.add(new JLabel("Right click backs out of a name"));
		help.add(new JLabel("Left and right click together selects a name"));
		help.add(new JLabel("Drag any name in or drag to reorder (useful if ordered highest priority closer to middle of window)"));
		help.add(new JLabel("Middle click runs def text as a command (if \"allow mindmap to run code\" in options)"));
		help.addSeparator();
		help.add(new JLabel("The + button adds a name, and - removes it."));
		help.addSeparator();
		help.add(new JLabel("You can search by parts of words like \"ord sea\" finds names containing both \"search\" and \"words\""));
		help.addSeparator();
		help.add(new JLabel("No save button. All changes are automaticly saved every "+ListwebRoot.varSaveInterval()/60+" minutes"));
		help.add(new JLabel("and copied to version history every "+ListwebRoot.vervarSaveInterval()/60+" minutes, when changes exist."));
		help.add(new JLabel("To get those versions, like if you delete or change by accident,"));
		help.add(new JLabel("they're the .jsonperline files in "+ListwebRoot.rootDir));
		help.add(new JLabel("The .json files are smaller and easier to read and are the newest of each name."));
		help.addSeparator();
		help.add(new JLabel(ListwebUtil.progName+" is opensource "+ListwebUtil.licenseName+" license."));
		help.add(new JLabel("To get the source code, unzip this jar file which you doubleclicked."));
		menu.add(help);
		final String btnTextDisabled = "Research>>", btnTextEnabled = "Research<<";
		final JMenuItem btnScreen = new JMenuItem(btnTextDisabled);
		btnScreen.addActionListener(new AbstractAction(btnTextDisabled){
			public void actionPerformed(ActionEvent e){
				btnScreen.setText(btnScreen.getText().equals(btnTextDisabled) ? btnTextEnabled : btnTextDisabled);
				toggleScreen.run();
			}
		});
		menu.add(btnScreen);
		return menu;
	}

}