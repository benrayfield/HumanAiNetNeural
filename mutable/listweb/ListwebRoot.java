package mutable.listweb;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.NavigableSet;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.*;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import immutable.occamsjsonds.*;
import mutable.util.Files;

public class ListwebRoot{
	private ListwebRoot(){}
	
	//static{System.err.println("TODO theres still mutable ops on NavigableMaps and Lists to change to immutable (using setWithEvent(Object...) and setWithoutEvent(Object...)). These are the keys prilist, def, and views. For now I'm leaving them as mutable and just not adding any more. I'm about to immutably add rbmParams key to some nodes to be used with rbmCircleUi. It has a few levels deep of params, and they will all be immutable past the rbmParams key. All expansions of this software must be immutable, and eventually I'll rewrite that mutable code so only the top level namespace is mutable, and maybe even that will be immutable using fork-editable treemap (BenRayfield's not to self, see ufnode/funode).");}
	
	/** map of mindmapItemName (unescaped filename) to json value in that file. Infoflow both directions.
	This is the only NavigableMap allowed to be mutable as its the namespace.
	*/
	public static final NavigableMap<String,NavigableMap<String,Object>> listweb =
		new TreeMap(JsonDS.mapKeyComparator);
	
	/** All names. The NavigableMap starts with few names because most are not loaded from harddrive.
	This SortedSet can only have names that dont exist if they're deleted since program last started.
	This is used for searching names without having to load them all from harddrive.
	This is a cache, so adding to this SortedSet is not an event that should fireListeners.
	<br><br>
	FIXME this synchronizedSet might slow things down too much. The search "all" is already slow before this.
	I'm doing this trying to fix...
	<br><br>
	Not scrolling to 0.0, already there
	saveChanges? countModified=5
	Exception in thread "AWT-EventQueue-0" java.util.ConcurrentModificationException
		at java.util.TreeMap$KeySpliterator.forEachRemaining(TreeMap.java:2753)
		at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:512)
		at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:502)
		at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:576)
		at java.util.stream.AbstractPipeline.evaluateToArrayNode(AbstractPipeline.java:255)
		at java.util.stream.ReferencePipeline.toArray(ReferencePipeline.java:438)
		at listweb.ui.TwoPrilistsPanel.command(TwoPrilistsPanel.java:284)
		at listweb.ui.TwoPrilistsPanel.onChange(TwoPrilistsPanel.java:110)
		at listweb.ui.TwoPrilistsPanel$1.insertUpdate(TwoPrilistsPanel.java:72)
		at javax.swing.text.AbstractDocument.fireInsertUpdate(AbstractDocument.java:201)
		at javax.swing.text.AbstractDocument.handleInsertString(AbstractDocument.java:748)
		at javax.swing.text.AbstractDocument.insertString(AbstractDocument.java:707)
		at javax.swing.text.PlainDocument.insertString(PlainDocument.java:130)
		at javax.swing.text.AbstractDocument.replace(AbstractDocument.java:669)
	Saving c:\temp\acyc\json\var\d\default_Down_Stack.json
		at javax.swing.text.JTextComponent.replaceSelection(JTextComponent.java:1371)
		at javax.swing.text.DefaultEditorKit$DefaultKeyTypedAction.actionPerformed(DefaultEditorKit.java:884)
		at javax.swing.SwingUtilities.notifyAction(SwingUtilities.java:1663)
		at javax.swing.JComponent.processKeyBinding(JComponent.java:2879)
		at javax.swing.JComponent.processKeyBindings(JComponent.java:2926)
		at javax.swing.JComponent.processKeyEvent(JComponent.java:2842)
		at java.awt.Component.processEvent(Component.java:6304)
		at java.awt.Container.processEvent(Container.java:2234)
		at java.awt.Component.dispatchEventImpl(Component.java:4883)
		at java.awt.Container.dispatchEventImpl(Container.java:2292)
		at java.awt.Component.dispatchEvent(Component.java:4705)
		at java.awt.KeyboardFocusManager.redispatchEvent(KeyboardFocusManager.java:1954)
		at java.awt.DefaultKeyboardFocusManager.dispatchKeyEvent(DefaultKeyboardFocusManager.java:806)
		at java.awt.DefaultKeyboardFocusManager.preDispatchKeyEvent(DefaultKeyboardFocusManager.java:1074)
		at java.awt.DefaultKeyboardFocusManager.typeAheadAssertions(DefaultKeyboardFocusManager.java:945)
		at java.awt.DefaultKeyboardFocusManager.dispatchEvent(DefaultKeyboardFocusManager.java:771)
		at java.awt.Component.dispatchEventImpl(Component.java:4754)
		at java.awt.Container.dispatchEventImpl(Container.java:2292)
		at java.awt.Window.dispatchEventImpl(Window.java:2739)
		at java.awt.Component.dispatchEvent(Component.java:4705)
		at java.awt.EventQueue.dispatchEventImpl(EventQueue.java:746)
		at java.awt.EventQueue.access$400(EventQueue.java:97)
		at java.awt.EventQueue$3.run(EventQueue.java:697)
		at java.awt.EventQueue$3.run(EventQueue.java:691)
		at java.security.AccessController.doPrivileged(Native Method)
		at java.security.ProtectionDomain$1.doIntersectionPrivilege(ProtectionDomain.java:75)
		at java.security.ProtectionDomain$1.doIntersectionPrivilege(ProtectionDomain.java:86)
		at java.awt.EventQueue$4.run(EventQueue.java:719)
		at java.awt.EventQueue$4.run(EventQueue.java:717)
		at java.security.AccessController.doPrivileged(Native Method)
		at java.security.ProtectionDomain$1.doIntersectionPrivilege(ProtectionDomain.java:75)
		at java.awt.EventQueue.dispatchEvent(EventQueue.java:716)
		at java.awt.EventDispatchThread.pumpOneEventForFilters(EventDispatchThread.java:201)
		at java.awt.EventDispatchThread.pumpEventsForFilter(EventDispatchThread.java:116)
		at java.awt.EventDispatchThread.pumpEventsForHierarchy(EventDispatchThread.java:105)
		at java.awt.EventDispatchThread.pumpEvents(EventDispatchThread.java:101)
		at java.awt.EventDispatchThread.pumpEvents(EventDispatchThread.java:93)
		at java.awt.EventDispatchThread.run(EventDispatchThread.java:82)
	Saving c:\temp\acyc\json\var\d\default_Up_Stack.json
	Saving c:\temp\acyc\json\var\s\start.json
	*/
	public static final SortedSet<String> namesCache = Collections.synchronizedNavigableSet(new TreeSet());
	//public static final SortedSet<String> namesCache = new TreeSet();
	
	/** backup prog limits filename (not path) to 220, so includes extensions
	such as ".jsonperline" (todo change that to .jsonl?).
	*/
	public static final int maxEscapedNameLen = 200; //TODO chars or bytes? Check at least Windows and Linux limits.
	
	/** Map of name to (last time saved to vervar). The var "last version" files are overwritten more often. *
	public static final NavigableMap<String,Double> cacheVervarModt = new TreeMap();
	*/
	
	/** TODO? Unlike the mindmap NavigableMap, metadata must be loaded for all mindmapItems,
	cached together in 1 file. It must be a cache, which means derivable from mindmap.
	Key is the same as in mindmap. Value is map with at least these keys:
	prilistListSize defStringSize
	TODO in later version, for maplists also include keys: econacycRelValue econacycCost.
	*
	public static final NavigableMap<String,NavigableMap<String,Object>> metadata = new TreeMap();
	*/
	
	/** Consumer<String> listens for any change of that mindmap name.
	FIXME instead of Set, use a SortedSet or TreeList since this software must be deterministic.
	*/
	private static final Map<String,Set<Consumer<String>>> mapStringToSetOfListener = new HashMap();
	
	private static final Map<Consumer<String>,Set<String>> mapListenerToSetOfString = new HashMap();
	
	/** names that need to be saved */
	private static final Set<String> modified = new HashSet<String>();
	
	/** Call this when change value of Root.mindmap key, including directly changing its deep contents.
	Does setModified then fireListeners. In a later version of this software,
	this will be obsolete because all data will be merkleForest which is immutable.
	*/
	public static synchronized void onChange(String name){
		pairWithItsFirstCharNameSoSearchCanFind(name); //causes another up to 2 onChanges if its not already paired, no infinite loop
		setModified(name);
		fireListeners(name);
	}
	
	private static synchronized void setModified(String name){
		namesCache.add(name); //TODO Whats the smallest set of places this could be and catch all the names?
		if(Debug.logModified) lg("modified: "+name);
		//setModified(name, Time.time());
		//NavigableMap map = get(name);
		//Double uiTime = (Double) map.get("uiTime");
		Double uiTime = (Double) getElseNull(name, "uiTime");
		double now = Time.now();
		if(uiTime == null || uiTime<now){
			//in case server receives version modified slightly out of sync so ahead of our time
			//map.put("uiTime", now);
			setWithoutEventAndWithoutMarkingAsModified(name, "uiTime", now);
		}
		modified.add(name);
	}
	
	private static synchronized void fireListeners(String name){
		namesCache.add(name); //TODO Whats the smallest set of places this could be and catch all the names?
		Set<Consumer<String>> set = listeners(name);
		if(Debug.logSetOfListenersBeforeFiringEvent) lg("fireListeners about "+name+" to "+set);
		for(Consumer<String> listener : set){
			listener.accept(name);
		}
	}
	
	/** Vervar, not var.
	TODO store each option in def of some mindmapItem like name "acycOptionVervarSaveInterval"?
	If so, make sure to use permisvec to limit who and what values it can change.
	Def is the right place since its the only textfield displayed by default.
	Also add a field defRecogFunc, which permisvec will protect similarly.
	defRecogFunc defines allowed defs. Example: ["range" 60 3600].
	Example ["regex" "[1-9][0-9]{0,3}"] or ["and" ["regex" ".*blah.*"] ["not" ["regex" ".*xxyy.*"]]].
	Those syntax and search modes were never added. TODO wait for ufbrowser integration with mindmap.
	*/
	public static double vervarSaveInterval(){
		return 60*15; //15 minutes
	}
	
	/** var, not vervar. */
	public static double varSaveInterval(){
		return 60; //1 minute
	}
	
	/** TODO in a future version of this software,
	each queuedFunc takes global state as param and returns next global state,
	but for now since that state is the mutable NavigableMap mindmap, param and return are null.
	TODO ForkJoinTask and ForkJoinPool?
	*/
	private static volatile Function queue = null;
	public static synchronized void queue(Function f){
		if(queue == null) queue = f;
		else queue = queue.andThen(f);
	}
	public static synchronized void runQueuedFuncs(){
		final Function runNow = queue;
		if(runNow != null){
			queue = null;
			runNow.apply(null); //TODO see comment on queue var
		}
	}
	
	public static final File rootDir = new File(mutable.util.Files.dataDir,"mm");
	//public static final File rootDir = new File(mutable.util.Files.dataDir,"listweb");
	//public static final File rootDir = new File("c:\\temp\\acyc"); //FIXME
	
	/** files named by Util.escapeName(anyShortString) *
	public static final File jsonRootDir = new File(rootDir,"json");
	*/
	
	/** For later version when mapacyc either replaces or optionally is used with the json mindmap *
	public static final File mapacycRootDir = new File(rootDir,"mapacyc")
	
	/** For later version when derive acyc64 (Torrent Like Acyc Part Packet, tlapp) from mapacyc *
	public static final File acyc64RootDir = new File(rootDir,"acyc64");
	*/
	
	/*
	public static final File jsonVarDir = new File(jsonRootDir,"var"); //vars of 1 value each
	//public static File jsonVervarDir = new File(jsonRootDir,"vervar"); //versioned vars
	//"TODO Is eventLog a vervar? Will I use vervar with json. Not directly, because I'm waiting on mapacyc for that, which is binary, but can put json in a maplist with key 'json'"
	//public static File jsonEventDir = new File(jsonRootDir,"event");
	//public static File jsonEventLog = new File(jsonEventDir,"eventLog.jsonperline");
	//private static final OutputStream streamToEventLog; //TODO verify: closed when jvm closes
	*/
	static{
		rootDir.mkdirs();
		System.out.println("listweb dir: "+rootDir);
		//jsonVervarDir.mkdirs();
		//System.out.println("vervarDir(mindmap version history): "+jsonVervarDir);
		//jsonEventDir.mkdirs();
		//System.out.println("eventObjectLog: "+jsonEventDir);
		/*try{
			streamToEventLog = new FileOutputStream(jsonEventLog,true);
		}catch (FileNotFoundException e){
			throw new Err(e);
		}
		System.out.println("eventLog(TODO for automatic recovery from data corruption): "+jsonEventLog);
		*/
	}
	
	//public static final File jsonLockFile = new File(jsonRootDir,"lockSoOnly1CopyOfTheProgCanBeOpenAtATime_writeLockingIsOnlyNeededForJsonVersionSinceZingAndBinacycVersionsWillUseImmutableMerkleData.lock");
	
	public static final String rootName = "start";
	
	/** Returns from memory if exists. Else loads from file. Else creates new. */
	public static synchronized NavigableMap<String,Object> get(String name){
		NavigableMap<String,Object> node = listweb.get(name);
		if(node == null){
			//boolean createInMemory = ignoreSaveCommandForTesting || !fileOfJsonVar(name).exists();
			File f = fileOfJsonVar(name);
			boolean createInMemory = !f.exists();
			//lg("In GET, file="+f+" not exist so will create in memory, name="+name);
			if(createInMemory){
				node = new TreeMap(JsonDS.mapKeyComparator);
				node.put("prilist", new ArrayList());
				node.put("def", "");
				node = Collections.unmodifiableNavigableMap(node);
				//FIXME also connect symmetricly with rootName
				//if(!ignoreSaveCommandForTesting) setModified(name);
				listweb.put(name, node);
				if(Debug.logModified) lg("modified by get: "+name);
				setModified(name);
			}else{
				load(name);
				node = listweb.get(name);
				saveVervarIfItsTimeAndVarModifiedAndVersioningEnabled(name); //happens less often than save var file, so prog often closes without this
			}
		}
		return node;
	}
	
	public static synchronized void setWithoutEvent(Object... pathAndValue){
		setWithoutEventAndWithoutMarkingAsModified(pathAndValue);
		setModified((String)pathAndValue[0]); //FIXME Why should a "without event" mark something as modified? What if auto save by time interval happens at the wrong time (is it synchronized?)?
	}
	
	public static synchronized void setWithoutEventAndWithoutMarkingAsModified(Object... pathAndValue){
		if(pathAndValue.length < 3) throw new RuntimeException(
			"pathAndValue.length=="+pathAndValue.length+" must be at least 3");
		listweb.put(
			(String)pathAndValue[0],
			(NavigableMap)JsonDS.jsonSet(
				get((String)pathAndValue[0]),
				JsonDS.newArrayWithFirstRemoved(pathAndValue)
			)
		);
	}
	
	public static synchronized void setWithEvent(Object... pathAndValue){
		setWithoutEvent(pathAndValue);
		onChange((String)pathAndValue[0]);
	}
	
	public static synchronized Object getElseNull(Object... pathAndValue){
		Object o = listweb.get(pathAndValue[0]);
		if(o == null) return null;
		return JsonDS.jsonGet(o, JsonDS.newArrayWithFirstRemoved(pathAndValue));
	}
	
	public static boolean nameExists(String name){
		return listweb.containsKey(name) || fileOfJsonVar(name).isFile();
	}
	
	public static synchronized List<String> allNamesWithTodoTimesSortedAscending(){
		loadAllFiles();
		List<String> namesWithTodoTime = new ArrayList();
		for(String name : namesCache){
			if(getTodoTime(name) != 0){
				namesWithTodoTime.add(name);
			}
		}
		Collections.sort(namesWithTodoTime, (String x, String y)->{
			double xTime = getTodoTime(x);
			double yTime = getTodoTime(y);
			if(xTime < yTime) return -1;
			if(xTime > yTime) return 1;
			return 0;
		});
		return Collections.unmodifiableList(namesWithTodoTime);
	}
	
	public synchronized static void saveChanges(){
		//if(ignoreSaveCommandForTesting) return;
		String m[] = modified.toArray(new String[0]);
		if(Debug.logModified) lg("saveChanges? modified: "+Arrays.asList(m));
		else lg("saveChanges? countModified="+m.length);
		modified.clear();
		NavigableSet<String> couldntSave = new TreeSet();
		try{
			for(String name : m){
				try{
					save(name);
				}catch(Exception e){
					lg("Couldnt save (1 or both files) "+name+" so putting it back in modified set to try soon");
					couldntSave.add(name);
				}
			}
		}finally{
			modified.addAll(couldntSave);
		}
	}
	
	private static final boolean mindmapBugFileContentsCachedByOsSoDoesntSaveThemWhenItShould = true;
	
	/** Saves Util.mindmap.get(name) to file */
	public static synchronized void save(String name){
		//if(ignoreSaveCommandForTesting) return;
		NavigableMap<String,Object> node = listweb.get(name);
		if(node == null) throw new RuntimeException("Name not exist in memory: "+name);
		File f = fileOfJsonVar(name);
		byte data[] = Text.stringToBytes(JsonDS.jsonString(node));
		if(!f.exists() || (mindmapBugFileContentsCachedByOsSoDoesntSaveThemWhenItShould || f.length() != data.length && !mutable.util.Files.bytesEqual(mutable.util.Files.read(f),data))){
			mutable.util.Files.write(data, f);
		}else{
			lg("Not saving var file for name="+name+" because file content equals memory");
		}
		saveVervarIfItsTimeAndVarModifiedAndVersioningEnabled(name);
	}
	
	private static void saveVervarIfItsTimeAndVarModifiedAndVersioningEnabled(String name){
		if(vervarEnabled(name)){
			//block of time theres allowed to be only 1 vervar save
			double now = Time.now();
			double blockStart = now - now%vervarSaveInterval();
			double v = lastVarSaveTime(name);
			double vv = lastVervarSaveTime(name);
			if(vv < blockStart && vv<v-5){
				File f = fileOfJsonVervar(name);
				String json = ListwebUtil.jsonToSingleLine(JsonDS.jsonString(eventViewOfVar(name)));
				byte data[] = Text.stringToBytes(json+"\r\n");
				mutable.util.Files.append(data, f);
				//lg("Saved version of "+name+" time="+Time.time());
			}
		}
	}
	
	/** -Infinity if never saved. Var, not vervar. */
	public static double lastVarSaveTime(String name){
		File f = fileOfJsonVar(name);
		return f.isFile() ? f.lastModified()*.001 : -1./0;
	}
	
	/** -Infinity if never saved. Vervar, not var. */
	public static double lastVervarSaveTime(String name){
		File f = fileOfJsonVervar(name);
		return f.isFile() ? f.lastModified()*.001 : -1./0;
	}
	
	public static synchronized NavigableMap<String,Object> load(String name){
		File f = fileOfJsonVar(name);
		if(f.isDirectory()) throw new Err("For name="+name+" need to create file="+f+" but its a dir");
		if(!f.isFile()) throw new RuntimeException("File not exist ["+f+"] for name["+name+"]");
		NavigableMap<String,Object> node = (NavigableMap<String,Object>) JsonDS.jsonParse(Text.bytesToString(mutable.util.Files.read(f)));
		namesCache.add(name); //TODO Whats the smallest set of places this could be and catch all the names?
		namesCache.addAll((List<String>)node.get("prilist")); //TODO Whats the smallest set of places this could be and catch all the names?
		Object prevValue = listweb.put(name,node);
		//lg("load("+name+") prevValue="+prevValue);
		if(prevValue != null){
			if(Debug.logModified) lg("modified by load: "+name);
			setModified(name);
		}
		fireListeners(name);
		return node;
	}
	
	/** creates empty Set<Consumer<String>> if not exist */
	protected static Set<Consumer<String>> listeners(String name){
		Set<Consumer<String>> set = mapStringToSetOfListener.get(name);
		if(set == null){
			set = new HashSet();
			mapStringToSetOfListener.put(name, set);
		}
		return set;
	}
	
	/** Creates empty Set<String> if not exist */
	protected static Set<String> listenees(Consumer<String> listener){
		Set<String> set = mapListenerToSetOfString.get(listener);
		if(set == null){
			set = new HashSet();
			mapListenerToSetOfString.put(listener, set);
		}
		return set;
	}
	
	/** Fires 1 event at listener for this name immediately to cover what was missed while not listening,
	and for all later changes.
	*/
	public static synchronized void startListening(Consumer<String> listener, String name){
		startListeningWithoutInstantEvent(listener, name);
		listener.accept(name); //first event
	}
	
	public static synchronized void startListeningWithoutInstantEvent(Consumer<String> listener, String name){
		if(Debug.logStartsAndStopsOfListening) lg("Start LISTEN to "+name+" WHO="+listener);
		listeners(name).add(listener);
		listenees(listener).add(name);
	}
	
	public static synchronized void stopListening(Consumer<String> listener, String name){
		if(Debug.logStartsAndStopsOfListening) lg("Stop LISTEN to "+name+" WHO="+listener);
		listeners(name).remove(listener);
		listenees(listener).remove(name);
		//TODO remove up to 1 empty set from both maps
	}
	
	public static synchronized void stopListening(Consumer<String> listener){
		String names[] = listenees(listener).toArray(new String[0]);
		for(String name : names) stopListening(listener, name);
	}
	
	public static synchronized void stopListening(String name){
		Consumer<String> listeners[] = listeners(name).toArray(new Consumer[0]);
		for(Consumer<String> listener : listeners) stopListening(listener, name);
	}
	
	public static File fileOfJsonVar(String name){
		//TODO merge duplicate code between fileOfJsonVar and fileOfJsonthenbinVar and fileOfBinVar
		return new File( new File(rootDir,ListwebUtil.escapeName(name.substring(0,1))),
			ListwebUtil.escapeName(name)+".json" );
	}
	
	/** not versioned. different file than fileOfJsonVar. *
	public static File fileOfJsonthenbinaryVar(String name){
		//TODO merge duplicate code between fileOfJsonVar and fileOfJsonthenbinVar and fileOfBinVar
		return new File( new File(rootDir,ListwebUtil.escapeName(name.substring(0,1))),
			ListwebUtil.escapeName(name)+".jsonthenbinary" );
	}*/
	
	/** not versioned. different file than fileOfJsonVar. *
	public static File fileOfBinaryVar(String name){
		//TODO merge duplicate code between fileOfJsonVar and fileOfJsonthenbinVar and fileOfBinaryVar
		return new File( new File(rootDir,ListwebUtil.escapeName(name.substring(0,1))),
			ListwebUtil.escapeName(name)+".binary" );
	}*/
	
	public static File fileOfJsonVervar(String name){
		return new File( new File(rootDir,ListwebUtil.escapeName(name.substring(0,1))),
			ListwebUtil.escapeName(name)+".jsonperline" );
	}
	
	/** returns true if changed */
	static boolean putAtTopOfPrilist_noEvent(String parent, String child){
		NavigableMap<String,Object> node = get(parent);
		List prilist = (List) node.get("prilist");
		int i = prilist.indexOf(child);
		boolean change = i!=0;
		if(change){
			if(Debug.logModified) lg("modified by putAtTopOfPrilist_noEvent: "+parent);
			List newPrilist = new ArrayList(prilist);
			if(i != -1) newPrilist.remove(i);
			newPrilist.add(0, child);
			setWithoutEvent(parent, "prilist", newPrilist);
			//setModified(parent); //FIXME Why should a "noEvent" mark something as modified? What if auto save by time interval happens at the wrong time (is it synchronized?)?
		}
		return change;
	}
	
	public static void unpairAll(String x){
		for(String other : prilist(x)){
			if(!other.equals(x)){
				unpair(other, x);
			}
		}
	}
	
	public static void deleteIncludingHistory(String x){
		Files.delete(fileOfJsonVervar(x));
		Files.delete(fileOfJsonVar(x));
	}
	
	public static void permDeleteIncludingMyHistoryButNotHistoryOfMeBeingInOthers(String x){
		unpairAll(x);
		deleteIncludingHistory(x);
	}
	
	/** Removes each from the other's prilist. Creates nothing even if 1 or both names dont exist. Does 1 or both events. */
	public static void unpair(String x, String y){
		boolean changedX = false, changedY = false;
		if(nameExists(x)){
			changedX = ListwebRoot.prilist(x).contains(y);
			if(changedX) setWithoutEvent(x, "prilist", JsonDS.removeValueFromList(ListwebRoot.prilist(x),y));
		}
		if(nameExists(y)){
			changedY = ListwebRoot.prilist(y).contains(x);
			if(changedY) setWithoutEvent(y, "prilist", JsonDS.removeValueFromList(ListwebRoot.prilist(y),x));
		}
		if(changedX) onChange(x);
		if(changedY) onChange(y);
		
		/* MUTABLE:
		boolean xRemovedY = nameExists(x) && prilist(x).remove(y);
		boolean yRemovedX = nameExists(y) && prilist(y).remove(x);
		if(xRemovedY) onChange(x);
		if(yRemovedX) onChange(y);
		*/
	}
	
	/** moves if exists. else adds */
	public static void putAtTopOfPrilist(String parent, String child){
		boolean changedParent = putAtTopOfPrilist_noEvent(parent, child);
		//child is parent
		boolean changedChild = addToEndOfPrilistIfNotExist_noEvent(child, parent);
		if(changedParent) fireListeners(parent);
		if(changedChild) fireListeners(child);
	}
	
	/** returns true if changed */
	static synchronized boolean addToEndOfPrilistIfNotExist_noEvent(String parent, String child){
		NavigableMap<String,Object> node = get(parent); //create if not exist, including prilist and def keys
		List prilist = (List) node.get("prilist");
		boolean change = !prilist.contains(child);
		if(change){
			//prilist.add(child);
			setWithoutEvent(parent, "prilist", prilist.size(), child);
			//if(!ignoreSaveCommandForTesting) setModified(parent);
			if(Debug.logModified) lg("modified by addToEndOfPrilistIfNotExist_noEvent: "+parent);
			setModified(parent);
		}
		return change;
	}
	
	/** Similar to putAtTopOfPrilist(name,getFirstCharName(name)) except neither is reordered when exists.
	TODO This may get slow if theres too many names starting with the same char,
	like urls "http..." start with h, but it should be ok up to at least 10k such urls, probably more.
	It wont waste much harddrive space because firstCharNames arent versioned,
	and the change of each name to add its firstCharName into its prilist will only happen once
	unless its removed then it gets added back. 
	*/
	public static void pairWithItsFirstCharNameSoSearchCanFind(String name){
		String firstCharName = getFirstCharName(name);
		addToEndOfPrilistIfNotExist(name, firstCharName);
		addToEndOfPrilistIfNotExist(firstCharName, name);
	}
	
	public static void addToEndOfPrilistIfNotExist(String parent, String child){
		boolean changedParent = addToEndOfPrilistIfNotExist_noEvent(parent, child);
		//child is parent
		boolean changedChild = addToEndOfPrilistIfNotExist_noEvent(child, parent);
		//if(changedParent) fireListeners(parent);
		//if(changedChild) fireListeners(child);
		if(changedParent) onChange(parent);
		if(changedChild) onChange(child);
	}
	
	public static List<String> prilist(String name){
		return (List<String>) get(name).get("prilist");
	}
	
	/*private static void setModified(String name, double time){
		mindmap.get(name).put("modt", time);
		modified.add(name);
	}*/
	
	/** WARNING: This does not set the others symmetricly.
	Example: A TwoPrilistsPanel.command search sets searchResults.prilist to the search results
	but does not add searchResults name symmetricly to the prilists of the many who are in those results.
	It could inTheory be used after setting the prilists of others symmetricly,
	but the events are easier to sync in other funcs that update both sides before both events.
	Could modify the event funcs to call ListwebRoot.vervarEnabled(String) to only have event
	for changing pair where both are enabled, so there would be no event for abc.prilist being updated
	to contain searchResults, but there would be an event for adding abc to searchResults.prilist
	since it would be hardcoded that way in the Set<String> used by ListwebRoot.vervarEnabled(String)
	which contains searchResults and a few other names.
	*/
	public static void setPrilistWithoutModifyingThePrilistsOfItsContents(String name, List<String> prilist){
		NavigableMap<String,Object> node = get(name); //create (with prilist and def keys) if not exist
		if(Debug.logModified) lg("modified by setPrilist: "+name);
		//setModified(name);
		//node.put("prilist", prilist);
		//fireListeners(name);
		setWithEvent(name, "prilist", prilist);
	}
	
	/** symmetric, adds name at end of every prilist in that prilist if not exist (addToEndOfPrilistIfNotExist) and sets that prilist */
	public static void setPrilist(String name, List<String> prilist){
		setPrilistWithoutModifyingThePrilistsOfItsContents(name, prilist);
		for(String peerName : prilist){
			addToEndOfPrilistIfNotExist(peerName, name);
		}
	}
	
	/** Gets the "todoTime" field of a name else 0 if it doesnt have one */
	public static double getTodoTime(String name){
		if(Debug.logGetTodoTime) lg("getTodoTime "+name);
		if(!ListwebUtil.isValidName(name)) return 0; //FIXME remove all my invalid names instead of letting this kind of logic spread
		Double todoTime = (Double) get(name).get("todoTime");
		return todoTime==null ? 0 : todoTime;
	}
	
	/** Sets the "todoTime" field of a name or if param is 0 then removes it */
	public static void setTodoTime(String name, double todoTime){
		double prev = getTodoTime(name);
		if(prev != todoTime){
			setWithEvent(name, "todoTime", todoTime==0 ? null : todoTime);
		}
	}
	
	/** sorts that name's prilist, moving everything with a todoTime first,
	and sorting by earliest todoTime first within those, and does not reorder the others
	since user probably spent lots of time choosing that order
	and just want to see what you've scheduled move to the front.
	*/
	public static void sortByTimeEtc(String name){
		System.err.println("sortByTime "+name);
		Comparator<String> sort = new Comparator<String>(){
			public int compare(String x, String y){
				double xt = ListwebRoot.getTodoTime(x);
				double yt = ListwebRoot.getTodoTime(y);
				if(xt == 0) xt = Double.MAX_VALUE/4;
				if(yt == 0) yt = Double.MAX_VALUE/4;
				if(xt < yt) return -1;
				if(xt > yt) return 1;
				return 0; //Dont reorder anything else
			}
		};
		List<String> mutablePrilist = new ArrayList(prilist(name));
		Collections.sort(mutablePrilist, sort);
		List<String> newPrilist = Collections.unmodifiableList(mutablePrilist);
		setPrilistWithoutModifyingThePrilistsOfItsContents(name, newPrilist);
	}
	
	public static String def(String name){
		return (String) get(name).get("def");
	}
	
	public static void setDef(String name, String def){
		NavigableMap<String,Object> node = get(name); //create (with prilist and def keys) if not exist
		/*node.put("def", def);
		fireListeners(name);
		setModified(name);
		*/
		setWithEvent(name, "def", def);
	}
	
	/** returns null if there is none,
	which you might use the default value of new Jsonthenbin(JsonDS.emptyMap,new byte[0]) if this returns null.
	*
	public static Jsonthenbinary jsonthenbinary(String name){
		//TODO cache the Jsonthenbin object? For now we only get file system caching
		return new Jsonthenbinary(Files.read(fileOfJsonthenbinaryVar(name)));
	}*
	
	public static void setJsonthenbinary(String name, Jsonthenbinary val){
		//TODO cache the Jsonthenbin object? For now we only get file system caching
		Files.write(val.data(), fileOfJsonthenbinaryVar(name));
	}*/
	
	/** Creates if not exist.
	The "views" field is a map of name2 (normally those in my prilist) to view options
	for seeing/editing name2 in context of name, such as name is a stack (JList)
	that comes with a prilist editor (JList) of the name selected in stack,
	and view options include scroll position and whats selected in name2.prilist.
	This allows the same name2 to be in multiple JLists at once,
	such as to view far parts of a big list and drag between them.
	*
	public static NavigableMap<String,NavigableMap<String,Object>> views(String name){
		NavigableMap n = get(name);
		NavigableMap v = (NavigableMap) n.get("views");
		if(v == null){
			n.put("views", v = Collections.emptyNavigableMap());
			fireListeners(name);
		}else{
			if(prilist(name).size()*2 < v.size()) removeViewsAboutNamesNotInPrilist(name);
		}
		return v;
	}*/
	
	public static void garbcolViewsIfTooMany(String name){
		NavigableMap v = (NavigableMap) getElseNull(name, "views");
		if(v != null && prilist(name).size()*2 < v.size()){
			removeViewsAboutNamesNotInPrilist(name);
		}
	}
	
	/*public static NavigableMap<String,Object> view(String from, String to){
		NavigableMap<String,NavigableMap<String,Object>> views = views(from);
		NavigableMap<String,Object> view = views.get(to);
		if(view == null){
			views.put(to, view = Collections.emptyNavigableMap());
			fireListeners(from); //Not an event for (name)to because its (name)from's data about (name)to
		}
		return view;
	}*/
	
	/** Similar to garbcol. This removes views that are not useful since they're not in prilist.
	This is called automaticly by views(String) funcs when theres more than twice as many views as prilist.
	*/
	public static void removeViewsAboutNamesNotInPrilist(String from){
		Set<String> p = new HashSet(prilist(from));
		//avoid infinite loop when views(String) calls this, by not calling views(String) here
		NavigableMap<String,NavigableMap<String,Object>> newViews = new TreeMap(JsonDS.mapKeyComparator);
		newViews.putAll((NavigableMap) get(from).get("views"));
		if(newViews != null){
			Iterator<String> iter = newViews.keySet().iterator();
			while(iter.hasNext()) if(!p.contains(iter.next())) iter.remove();
		}
		setWithEvent(from, "views", Collections.unmodifiableNavigableMap(newViews));
	}	
	
	/** All names must connect to the name that is their first char, so they all are reachable at
	least 1 way. This func scans dirs in Root.jsonVarDir for names not yet in their first letter,
	such as "hello" and "h" in eachother's prilists. If x is in y's prilist, then y must be in x's prilist.
	The "%" dir contains names whose first char is escaped, including if their first char is actually "%".
	Also updates Root.rootName to contain all those chars. All updates keep their position if exist.
	*/
	public static void updateRootChars(){
		lg("Start updateRootChars reading filenames");
		SortedSet<String> rootCharNames = new TreeSet();
		for(String name : listweb.keySet().toArray(new String[0])){
			String rootCharName = name.substring(0,1);
			rootCharNames.add(rootCharName);
			addToEndOfPrilistIfNotExist(rootCharName, name);
		}
		for(File dir : rootDir.listFiles()){
			if(dir.isDirectory()){
				String filenames[] = dir.list();
				Arrays.sort(filenames); //this software must be deterministic
				for(String filename : filenames){
					if(filename.endsWith(".json")){
						String name = ListwebUtil.unescapeName(filename.substring(0, filename.length()-".json".length()));
						if(name.length() != 0){ //FIXME why is there a length 0 name, and why are there 2 dirs on my harddrive named that?
							String rootCharName = name.substring(0,1);
							rootCharNames.add(rootCharName);
							//System.err.println("This is causing everything to be marked as modified since we havent loaded all the files (nor should we) so connecting them creates new prilist and def, which probably causes problem when GET them later since wont be loaded from file. This need some redesign.");
							addToEndOfPrilistIfNotExist(rootCharName, name);
						}
					}
				}
			}			
		}
		for(String rootCharName : rootCharNames){
			addToEndOfPrilistIfNotExist(ListwebRoot.rootName, rootCharName);
			putAtTopOfPrilist("firstCharNames", rootCharName);
		}
		putAtTopOfPrilist(ListwebRoot.rootName, "firstCharNames");
		lg("End updateRootChars reading filenames");
	}
	
	/** 1-2 chars depending if its a unicode surrogate pair */
	public static boolean isFirstCharName(String name){
		if(name.length() == 1){
			if(Character.isSurrogate(name.charAt(0))) throw new Err(
				"Unclosed unicode low surrogate: (char)"+(int)name.charAt(0));
			return true;
		}else if(name.length() == 2){
			//FIXME updateRootChars and escapes dont work with this yet
			//FIXME Check for surrogates in wrong order or not matching other places in name,
			//such as high then low, and other places in this software.
			return Character.isLowSurrogate(name.charAt(0))
				&& Character.isHighSurrogate(name.charAt(1));
		}else{
			return false;
		}
	}
	
	/** 1-2 chars depending if its a unicode surrogate pair */
	public static String getFirstCharName(String name){
		return Character.isSurrogate(name.charAt(0)) ? name.substring(0,2) : name.substring(0,1);
	}
	
	public static synchronized void updateNameCacheFromContentsOfFirstChars(){
		prilist(rootName); //firstCharNames should be in rootName. Put them in namesCache
		for(String name : namesCache.toArray(new String[0])){
			if(isFirstCharName(name)){
				//Fast because normally loads file for name, but not the names in its prilist
				namesCache.addAll(prilist(name));
			}
		}
	}
	
	/*public static void set(String name, String keyInName, Object valueOrNull){
		NavigableMap m = get(name);
		//Some funcs rely on this func to call listeners every time, so caller should check if value changed if(Util.equals(m.get(keyInName),valueOrNull)) return; //set x to x
		m.put(keyInName, valueOrNull);
		fireListeners(name);
	}*/
	
	/** For saving to that name's vervar file. Its [time, name, value]. */
	public static List eventViewOfVar(String name){
		return Arrays.asList(Time.now(), name, get(name));
	}
	
	private static final Set<String> vervarOtherExclusions = Collections.synchronizedSet(new HashSet());
	
	/** Versioning is done on almost all vars/names, but not those which are used as caches,
	normally excluding isFirstCharName(String) and "searchResults" and "defaultUpStack" and "defaultDownStack" etc.
	*/
	public static void vervarDisable(String dontVersionThisName){
		vervarOtherExclusions.add(dontVersionThisName);
	}
	
	public static boolean vervarEnabled(String name){
		return !isFirstCharName(name) && !vervarOtherExclusions.contains(name);
	}
	
	public static void sortPrilist(String name, Comparator<String> c){
		//TODO When mapacyc upgrade, everything will be immutable so will have to replace it instead of sorting mutable
		Collections.sort(prilist(name), c);
		onChange(name);
	}
	
	/*public static void appendToEventLogForName(String name){
		NavigableMap node = get(name);
		NavigableMap event = new TreeMap();
		event.put("time", Time.time()); //will be sorted before val, so near start of each line.
		event.put("key", name);
		event.put("val", node);
		//TODO If ZingJson supported null, setting val to null could mean delete (should it?),
		//but in this early version of the software there are no deletes.
		//Should the lack of a val key mean its deleted?
		appendToEventLog(event);
	}
	
	/*public static void appendToEventLog(NavigableMap jsonMap){
		String json = ZingJson.toJson(jsonMap);
		json = Util.jsonToSingleLine(json);
		appendLineToEventLog.accept(json);
	}
	
	private static final Consumer<String> appendLineToEventLog = (String line)->{
		if(line.contains("\r") || line.contains("\n")) throw new Err("Multiline["+line+"]");
		byte b[] = Text.stringToBytes(line);
		try{
			streamToEventLog.write(b);
			streamToEventLog.write('\r');
			streamToEventLog.write('\n');
		}catch(Exception e){
			throw new Err(e);
		}
	};*/
	
	/** Slow. Only needed during maintenance and renames.
	TODO in a later version, if total files get too big,
	divide into n subsets and only have 2 in memory at a time.
	*/
	public static void loadAllFiles(){
		updateRootChars();
		for(String name : listweb.keySet().toArray(new String[0])){
			get(name);
		}
	}
	
	public static void main(String[] args){
		String s = "$~x\\/:;\r\nhello";
		String s2 = ListwebUtil.unescapeName(ListwebUtil.escapeName(s));
		System.out.println("equals?="+s.equals(s2));
		listweb.put("test xyz", (NavigableMap) JsonDS.jsonParse("{\"prilist\":[\"abc\", \"def\", 32.4], \"def\": \"blah blah\"}"));
		save("test xyz");
		System.out.println(load("test xyz"));
	}
	
	public static void clearStack(String stackName, String nameShouldBeAtStackFloor){
		List<String> backedStack = prilist(stackName);
		for(int i=backedStack.size()-1; i>0; i--){
			String name = backedStack.get(i);
			if(!name.equals(nameShouldBeAtStackFloor)){
				unpair(stackName, name);
			}
		}
		putAtTopOfPrilist(stackName, nameShouldBeAtStackFloor);
	}
	
	private static boolean booted;
	
	/** Repeated calls in same JVM do nothing, but it correctly throws if run the program twice at once in same dir. */
	public static synchronized void boot(){
		if(!booted){
			booted = true;
			/*if(ListwebRoot.jsonLockFile.exists()) throw new Err(
				"A copy of the prog is already running, which I know because file exists"
				+" (If this is not true, such as could not delete it while closing last time,"
				+" delete it and try again): "+ListwebRoot.jsonLockFile);
			try{
				ListwebRoot.jsonLockFile.createNewFile();
			}catch(IOException e){ throw new Err(e); }
			ListwebRoot.jsonLockFile.deleteOnExit(); //unlock jsonDir when this prog closes
			*/
			if(!Debug.skipRootCharsUpdateOnBootForSpeed){
				ListwebRoot.updateRootChars();
				ListwebRoot.saveChanges();
			}else{
				ListwebRoot.updateNameCacheFromContentsOfFirstChars();
			}
			System.out.println("root prilist: "+ListwebRoot.prilist(ListwebRoot.rootName));
			new Thread(){
				public void run(){
					while(true){
						ListwebRoot.saveChanges();
						double now = Time.now();
						double blockSize = ListwebRoot.varSaveInterval();
						double blockStart = now - now%blockSize;
						double next = blockStart+blockSize;
						Time.sleepNoThrow(next-now);
					}
				}
			}.start();
		}
	}
	
	static volatile boolean startedClosing;
	
	private static final Object inCaseTheEarlierGrimReaper_whoWhoMayHaveMoreAccessToFilesEtc_fails = new Object(){
		protected void finalize() throws Throwable{
			onClosingProg();
		}
	};
	
	public static synchronized void onClosingProg(){
		if(startedClosing) return;
		startedClosing = true;
		lg("Closing "+ListwebUtil.progName);
		try{
			while(!modified.isEmpty()){ //if save fails, keep trying forever so person may come find the problem and solve it instead of losing their data
				try{
					saveChanges();
				}catch(Throwable t){
					t.printStackTrace(System.err);
					lg("Failed to save all, so cant close prog, time="+Time.now());
				}
			}
		}finally{
			//if(jsonLockFile.isFile()) jsonLockFile.delete(); //in case kill prog instead of letting it die on its own
		}
	}

	/** Reduce !hdwmy to !hdwm, and replace !hd to !h, etc, but only where def starts with that. */
	public static void reduceHdwmy(){
		loadAllFiles();
		for(String name : listweb.keySet().toArray(new String[0])){
			String def = def(name);
			if(def.startsWith("!hdwmy")){
				setDef(name, "!hdwm"+def.substring(6));
			}else if(def.startsWith("!hdwm")){
				setDef(name, "!hdw"+def.substring(5));
			}else if(def.startsWith("!hdw")){
				setDef(name, "!hd"+def.substring(4));
			}else if(def.startsWith("!hd")){
				setDef(name, "!h"+def.substring(3));
			}
		}
	}

}