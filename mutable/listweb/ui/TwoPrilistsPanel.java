package mutable.listweb.ui;

import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.WeakHashMap;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Stream;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import mutable.listweb.ListwebRoot;
import mutable.listweb.ListwebUtil;
import mutable.listweb.Options;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;

/** auto saves changes when closed, so make sure to keep backups often. */
public class TwoPrilistsPanel extends JPanel{

	public final PrilistStack up, down;

	protected final JTextField textSearch;

	protected final JComboBox textContext;

	// protected final JCheckBox chkAlsoSearchDefs;

	protected boolean regexErr;

	/*
	 * "Will mindmap break (considering I designed the events) if editing or viewing the prilist of its own stack?"
	 * "TODO PrilistStack must use 2 prilists as its stack, or maybe each name can have a prilist and prilistGhost. Also, how to store which was selected in a prilist to restore it?"
	 * "Should a prilist's stack display its top item? If its always prilistPrilist, dont need to, but if its potentially different each time, like rightclick menu on different objects, then would store many such stacks."
	 * "Need the ability to view the same prilist in 2 different prilistStacks at 2 different scrollPositions"
	 * "scrollPositions should be either integer or string, for how far up/down the stack to view, from screen center considering that prilistStackUi can be up or down. This will as long as the scroll positions are stored separately per prilistStack. But I dont want to store extremely many scroll positions in each stack since any name can be a stack."
	 */

	public final String nameOfSearchResults;

	public TwoPrilistsPanel(final String topStackName, String bottomStackName, String selectPtr,
			String nameOfSearchResults){
		this.nameOfSearchResults = nameOfSearchResults;
		setLayout(new GridLayout(2, 1));
		up = new PrilistStack(topStackName, selectPtr, true);
		down = new PrilistStack(bottomStackName, selectPtr, false);
		JPanel upPanel = new JPanel(new BorderLayout());
		textSearch = new JTextField();
		textSearch.getDocument().addDocumentListener(new DocumentListener(){
			public void insertUpdate(DocumentEvent e){
				onChange("search");
			}

			public void removeUpdate(DocumentEvent e){
				onChange("search");
			}

			public void changedUpdate(DocumentEvent e){
			}
		});
		textContext = new JComboBox(new String[] { "name/fast", "times", "all/time", "regex/time", "name/time", "command" });
		ActionListener a = new ActionListener(){
			public void actionPerformed(ActionEvent e){
				onChange("context");
			}
		};
		textContext.addActionListener(a);
		// chkAlsoSearchDefs = new JCheckBox();
		// chkAlsoSearchDefs.setToolTipText("also search defs (if \"find all\"
		// or \"regex\")");
		// chkAlsoSearchDefs.addActionListener(a);
		JPanel p = new JPanel(new BorderLayout());
		p.add(textSearch, BorderLayout.CENTER);
		JPanel midRight = new JPanel(new BorderLayout());
		// midRight.add(textContext, BorderLayout.CENTER);
		// midRight.add(chkAlsoSearchDefs, BorderLayout.EAST);
		// p.add(midRight, BorderLayout.EAST);
		p.add(textContext, BorderLayout.EAST);
		upPanel.add(p, BorderLayout.SOUTH);
		upPanel.add(up, BorderLayout.CENTER);
		add(upPanel);
		add(down);
	}
	
	protected boolean isRecursingIstimes;

	/** whatChanged is "search" or "context" and may have other values in future versions of this software */
	protected void onChange(String whatChanged){
		String s = textSearch.getText();
		String context = (String) textContext.getSelectedItem();
		boolean isTimes = context.equals("times"); //No query if by "times"
		//if(isTimes) SwingUtilities.invokeLater(()->textSearch.setText(""));
		if(isTimes && !isRecursingIstimes){
			isRecursingIstimes = true;
			if(whatChanged.equals("search")) {
				textContext.setSelectedItem("name/fast");
			}else if(whatChanged.equals("context")){
				textSearch.setText("");
			}
			isRecursingIstimes = false;
			return;
		}
		
		boolean searchInDefs = Options
				.option(ListwebUtil.optionSearchInDefs, false);
		final String out = command(textSearch.getText(), context, searchInDefs);
		final int c = textSearch.getCaretPosition();
		SwingUtilities.invokeLater(()-> {
			textContext.setBackground(regexErr ? Color.gray : Color.white);
			if (!s.equals(out)) {
				textSearch.setText(out);
				textSearch.setCaretPosition(Math.min(c, out.length()));
			}
		});
	}

	/**
	 * when middleclick a name, runs it as a command, which is url if matches
	 * that, open if file or dir exists, else use its def as command. Does
	 * nothing if options dont allow that.
	 */
	public static void runNameAsCommand(String name){
		if (Options.option(ListwebUtil.optionAllowRunCodeInDefs,
				false)) {
			String command;
			if (Text.isURL(name))
				command = "url " + name;
			else if (new File(name).exists())
				command = "open " + name;
			else
				command = ListwebRoot.def(name);
			command(command);
		}
	}

	/**
	 * the "command" context, a subset of the more general
	 * command(commandOrString,context,alsoSearchDefs). This is also callable by
	 * middleclick a name to run its def as command.
	 */
	public static String command(String command){
		lg("Running command: " + command);
		try {
			String tokens[] = Text.splitByWhitespaceNoEmptyTokens(command);
			if (tokens.length < 2)
				return ""; // no single word commands exist yet
			if (tokens[0].equals("url")) {
				Desktop.getDesktop().browse(new URI(tokens[1]));
			} else if (tokens[0].equals("open")) {
				Desktop.getDesktop().open(new File(tokens[1]));
			} else if (tokens[0].equals("edit")) {
				Desktop.getDesktop().edit(new File(tokens[1]));
			}
			return command;
		} catch (Exception e) {
			throw new Err(e);
		}
	}
	
	//private static final WeakHashMap<NavigableMap,String> searchCache = new WeakHashMap();
	//FIXME this will run out of memory eventually but so slowly its probably ok cuz
	//its only data created by Humans that adds to memory, including a copy of prilist and def
	//at potentially every change. Wanted WeakHashMap but need == instead of equals. TODO get both.
	private static final Map<NavigableMap,String> searchCache = new IdentityHashMap();
	
	static boolean findAll_matches(String[] tokensLowercase, String name){
		//Much slower the first time than the other kinds of search cuz
		//loads all the names, last version only. Could store them in bigger less files or a database,
		//but that would be a big redesign, and I might wait for bigger redesign using ufnode/ufmaplist.
		//First GET per name is from separate file on harddrive.
		NavigableMap value;
		try{
			if(!ListwebUtil.isValidName(name)) return false; //avoid throw
			value = ListwebRoot.get(name);
		}catch(Exception e){ //2017-9-15 Its been failing on "EscapedName too long" so I put this try/catch
			e.printStackTrace(System.err);
			return false;
		}
		String nameConcatDefAllLowercase = searchCache.get(value);
		if(nameConcatDefAllLowercase == null){
			nameConcatDefAllLowercase = (name+" "+value.get("def")).toLowerCase();
			searchCache.put(value, nameConcatDefAllLowercase);
		}
		/*if(nameConcatDefAllLowercase == null){
			List<String> prilist = (List<String>) value.get("prilist");
			String def = (String) value.get("def");
			StringBuilder sb = new StringBuilder(name.toLowerCase());
			if(prilist != null) for(String inPrilist : prilist){
				sb.append(' ').append(inPrilist.toLowerCase());
			}
			if(def != null) sb.append(def.toLowerCase());
			searchCache.put(value, nameConcatDefAllLowercase=sb.toString());
		}*/
		for(String token : tokensLowercase){
			if(!nameConcatDefAllLowercase.contains(token)) return false;
		}
		return true;
	}

	/** returns the text to replace the command as you type it.
	FIXME TODO this should be done in a "command" property of some mindmapItem,
	such as the up stack or another ptr in constructor, and let events cause the command
	whenever the text changes, which may replace that someptr.command therefore update display in ui.
	FIXME TODO Similar for each AddTextRemove?
	*/
	protected String command(String s, String context, boolean alsoSearchDefs){
		if(alsoSearchDefs) return "uncheck option search in defs";
		regexErr = false;
		boolean isFastFindName = context.equals("name/fast"); //doesnt sort by todoTimes so doesnt load values
		boolean isFindName = context.equals("name/time");
		boolean isFindAll = context.equals("all/time");
		//boolean isFindAll = context.equals("name,list,def");
		boolean isRegex = context.equals("regex/time");
		boolean isTimes = context.equals("times");
		boolean isCommand = context.equals("command");
		if(isCommand){
			return command(s);
		}
		if(s.endsWith("  ")) return "";
		if(s.trim().length() > 1 || isTimes){
			final String tokensLowercase[] = Text.splitByWhitespaceNoEmptyTokens(s.toLowerCase());
			
			Predicate<String> hardQuery = null;
			if(isTimes){ //find all names that have a todoTime. No query. All of them.
				hardQuery = (String name)->ListwebRoot.getTodoTime(name)!=0;
			}else if(isFastFindName || isFindName){
				hardQuery = (String name)->{
					name = name.toLowerCase();
					for(String token : tokensLowercase){
						if(!name.contains(token.toLowerCase())) return false;
					}
					return true;
				};
			}else if(isFindAll){
				hardQuery = (String name)->{
					//moved it to a function cuz Eclipse debugger is
					//saying conditional breakpoint doesnt compile (2019-11-25)
					//even though it does.
					return findAll_matches(tokensLowercase,name);
				};
			}else if(isRegex){
				try{
					final Pattern regex = Pattern.compile(s);
					hardQuery = (String name)->{
						return regex.matcher(name).matches();
					};
				}catch(PatternSyntaxException e){
					regexErr = true;
				}
			}	
			else throw new Todo("Unknown search/command context: "+context);
			
			Comparator<String> sort = new Comparator<String>(){
				public int compare(String x, String y){
					if(!isFastFindName){
						double xt = ListwebRoot.getTodoTime(x);
						double yt = ListwebRoot.getTodoTime(y);
						/*if(xt == 0){
							if(yt == 0){
								//sort by length of name
							}else{
								//FIXME backward?
								return -1; //if have todoTime its before all that dont
							}
						}else{
							if(yt == 0){
								//FIXME backward?
								return 1; //if have todoTime its before all that dont
							}else{
								if(xt == yt){
									//sort by length of name, both before those without todoTime
								}else{
									if(xt < yt) return 1; //FIXME backward?
									if(xt > yt) return -1; //FIXME backward?
									return 0;
								}
							}
						}*/
						if(xt == 0) xt = Double.MAX_VALUE;
						if(yt == 0) yt = Double.MAX_VALUE;
						if(xt < yt) return -1;
						if(xt > yt) return 1;
					}
					if(x.length() < y.length()) return -1; //prefer shorter
					if(x.length() > y.length()) return 1;
					//if same len (and todoTime comparing rules), compare strings as usual
					return x.compareTo(y);
				}
			};
			if(hardQuery != null){
				List<String> searchResults;
				synchronized(ListwebRoot.class){ //namesCache is synchronized other places on ListwebRoot.class in static synchronized funcs
					//searchResults = ListwebRoot.namesCache //instead of ListwebRoot.listweb.keySet()
						//.stream()
						//.filter(hardQuery)
						//.sorted(sort)
						//.toArray((int siz)->{return new String[siz];});
						//commented those cuz ConcurrentModificationException. These streams must be doing it while toArray.
					String[] allNames = ListwebRoot.namesCache.toArray(new String[0]);
					//Stream allNames = Arrays.asList(ListwebRoot.namesCache.toArray(new String[0])).stream();
					//searchResults = (String[]) allNames.filter(hardQuery).sorted(sort)
					//	.toArray((int size)->new String[size]);
					searchResults = new ArrayList();
					for(int i=0; i<allNames.length; i++){
						String name = allNames[i];
						if(hardQuery.test(name)){
							searchResults.add(name);
							lg("Search result["+name+"] query["+s+"] remainingToSearch="+(allNames.length-1-i));
						}
					}
					Collections.sort(searchResults,sort);
					searchResults = Collections.unmodifiableList(searchResults);
				}
				ListwebRoot.setPrilistWithoutModifyingThePrilistsOfItsContents(nameOfSearchResults, searchResults);
				boolean popGhosts = !ListwebRoot.prilist(up.stackName).contains("searchResults");
				up.acycSelectInStack("searchResults", popGhosts);
				//up.select("searchResults", false);
				//System.out.println("Search results: "+Arrays.asList(searchResults));
			}
		}
		return s;
	}

}