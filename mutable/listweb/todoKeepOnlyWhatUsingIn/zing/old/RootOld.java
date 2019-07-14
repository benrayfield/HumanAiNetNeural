package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.WeakHashMap;

/** root namespace (normally 1 per computer) mapping between hashname (such as h$hexSha256...) and json object.
The kind of mapping is weak, identity, and thread-safe.
<br><br>
TODO correct behavior of hashname and jsonOb:
<br><br>
StrongReachable jsonOb not yet hashed does not perSe create the hashname
<br><br>		
StrongReachable jsonOb thats ever been hashed prevents garbcol of that hashname.
<br><br>
StrongReachable hashname without its jsonOb in memory does nothing.
The jsonOb may be on harddrive or internet waiting to be loaded.
<br><br>
StrongReachable hashname with its jsonOb in memory prevents garbcol of that jsonOb.
<br><br>
TODO New NavigableMaps and Lists should prefer the hashname if it exists,
but checking if it exists requires synchronized so most times wont be done,
and instead derive the hashed forms later. Or maybe they'll all point at Node
which has NavigableMap/List and a place to store hash,
so new collections arent needed when things get hashed.
But Node can be dup since multiple of them can be created
and have the same hash and later optionally dedup,
so Node containing econacyc vars is complicated by those dup.
Also, nodes on harddrive but not in memory may need to be included in econacyc.  
*/
public class RootOld{
	private RootOld(){}
	
	//"TODO should there be some kind of metadata object for each jsonOb to hold things like hashname and econacycCost and econacycRelValue?"
	
	/** root namespace. keys are both hashname and json object, mapping both directions *
	public static final Map ns = Collections.synchronizedMap(todoWeakIdentityMap);
	*/
	
	//static final Map<String,Object> hashToOb = Collections.synchronizedMap(new WeakHashMap());
	static final Map<String,Node> hashToOb = new WeakHashMap();
	
	/*
	//static final Map<Object,String> obToHash = Collections.synchronizedMap(weakIdentityMap);
	static final Map<Node,String> obToHash = "weakIdentityMap of jsonOb to weakReference<hashname>";
	static final Map<Node,String> obToHash = "weakIdentityMap of jsonOb to hashname";
	*/
	
	/*"TODO Since forest may have jsonOb or hashname as child pointers, but need to keep one if have the other (or which combinations?), how will that work?"
	
	"If any strongReachable jsonOb points at hashname, then both that hashname and 1 forestEqual jsonOb will be in hashToOb and therefore obToHash's key exists"
	
	"If any strongReachable hashname exists, "
	
	...Correct behavior:
	
	"strongReachable jsonOb not yet hashed does not perSe create the hashname"
		
	"strongReachable jsonOb thats ever been hashed prevents garbcol of that hashname"
		
	"strongReachable hashname without its jsonOb in memory does nothing"
		
	"strongReachable hashname "
	*/
		
		
	
	/*static synchronized void add(String hash, Object ob){
		hashToOb.put(hash, ob);
		//obToHash.put(ob, hash);
	}*/
	
	public static synchronized Object obOrNull(String hash){
		//return hashToOb.get(hash);
		Node n = hashToOb.get(hash);
		if(n == null) return null;
		return n.ob;
	}
	
	/*public static synchronized String hash(Object ob){
		/*String h = obToHash.get(ob);
		if(h == null) h = directHash(ob);
		return h;
		*
		
		/*Node n = hashToOb.get(key)
		if(n == null) return null;
		return n.hash();
		*
	}*/
	
	//"TODO where to dedup nodes?"
	//"TODO where to dedup nodes? And what about econacyc? and how are references counted? Or are they counted?"
	
	public static String[] hashes(){
		String h[] = hashToOb.keySet().toArray(new String[0]);
		Arrays.sort(h);
		return h;
	}
	
	/** do the hash instead of checking the Map.
	TODO generalize to multiple possible hash algorithms, or just use sha256 of base64 for now? 'h$base64Ofsha2SuchAsSha256...'
	*
	static String directHash(Object ob){
		
	}*/
	
	/** immutable snapshot of state, usually not yet (and may never be) hashed because of lazyEvalHash.
	This prevents garbcol of the hashes and objects reachable from it, which otherwise may be garbcol because weakReference.
	*/
	static NavigableMap state = Collections.emptyNavigableMap();
	
	/** immutable state */
	public static void setState(NavigableMap state){
		RootOld.state = state;
	}
	
	/** immutable state */
	public static NavigableMap getState(){
		return state;
	}

}
