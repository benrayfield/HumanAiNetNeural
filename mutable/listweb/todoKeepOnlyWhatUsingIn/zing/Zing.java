package mutable.listweb.todoKeepOnlyWhatUsingIn.zing;

import java.util.Collections;
import java.util.Iterator;
import java.util.Map;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.LongSize;

/** immutable lazyEval merkle forest of map, list, and bitstring,
where any bitstring can be read as any primitive at any bit index,
and can cache java lambdas (of various primitive andOr object types)
as compiled from those strings of java code,
but only those allowed by a whitelist of core code strings
and anything derived from them, so it can be proven sandboxed
or used in a non-sandboxed way.
<br><br>
This can be done in ways compatible with java debuggers such as Eclipse and Netbeans,
at least by precompiling the lambdas in java as usual and only verifying the hash
of that code matches when use the hash to get an instance of those lambdas,
but I'm hoping it can be done dynamicly with javassist
(using java8 lambda classes but not that syntax) and create those files at runtime
if using a debugger so lambda code created at runtime would still be debuggable.
Maybe create eclipse andOr netbeans plugin to do that if necessary,
but I'm hoping just javassit and creating the .java and .class files at runtime
will be enough to make the debugger work.
Does javassist add line numbers to bytecode?
<br><br>
Also I'm considering making a version on android and javascript.
I may use translation of some parts of java code to javascript,
or I may compile separately of those languages
so javascript would only run in browsers and maybe java script support etc,
but I'd rather standardize everything to java syntax and translate.
<br><br>
???
figured out == vs .equals..."
Zings are always equal by bitstring, and they're kept small enough to compare them that way when used as keys.
They dont recurse in .equals like java Collections, so its ok to compare by content if they're all hashed.
But most zings will be lazyEval hashed, so it has to be == if they're used as keys.
Calling equals should trigger lazyEval hash and should be done mostly for keys, doing it for values only as needed.
So its decided, zings are by .equals which triggers lazyEval hash recursively.
<br><br>
TODO should Root use weakref? For now, no, make them garbcolable by my own code since I want lazyEvalHash.
<br><br>
zingSizeSizeBit.
zingIsMapBit.
zingIsListBit.
zingIsHashBit.
listSize is 12 or 60 bits depending on zingSizeSizeBit. If leaf, its 0.
bitSize is 16 or 64 bits depending on zingSizeSizeBit. This is of the whole zing including header.
if(zingIsMapBit || zingIsListBit){
	remaining cumulativeBitSizes descending, starting at the next lower than bitSize. Always listSize-1 of these. Same 16 or 64 headerIntegerSize as bitSize.
}
content is concat of all in the list or the content of the leaf. Content is hash concat zingHashType if zingIsHashBit.
<br><br>
As Comparable<Zing>, its first by length (in case leading 0s) then as unsigned integer.
<br><br>
equals(Object) must answer the same as compareTo(Object) == 0.
<br><br>
TODO rewrite this without the TODOs after decide on a design I was figuring out here:
hashCode() must not call hash(byte) because its lazyEvalHash on the slow secureHash,
instant int32 hash. Make sure not to create childs before the int32 hash if the
implementation does lazyEvalCreateChild. Instead, make that a lazyEvalHashInt32,
which is useful when constructing with a bitstring instead of existing childs.
There are 4 core types of zing: leaf, map, list, hash.
There should be either 3 or 4 kinds of hashInt32,
because I'm undecided (TODO) if hash should have its own hashInt32
or use the hash of the thing its of. It would complicate the datastructs if
had to know the hashInt32 of the content that generated a hash
since would have to go get that or store it in an extra 32 bits in hash.
If the math is consistent for hash to have a different hashInt32 than its content,
then I'll choose that design. TODO Is that math consistent?
The math is consistent because theres a deterministic way to choose
for a child c to be literal or its hash (and that is Z.ptr(Zing))
therefore hash and c having different hashInt32
will never cause different parent hash.
<br><br>
Therefore need 4 algorithms for hash32 of leaf, map, list, and hash,
and these should all be perfectHashing which means the algorithm is
generated from random data, in this case once per JVM run,
since java only requires hashInt32 to be the same per run of JVM.
This combined hash algorithm (depending which type of Zing) is in Z.hash32(Zing).
<br><br>
hash32 of leaf or hash:
wrapped sum of each Zing.ja(i)^randArray[i], then xor the 2 halfs of the long into an int32.
Once per JVM run, an array of int64 is created,
size random (suggested in range 100-200), of random int64s.
<br><br>
hash32 of map or list:
This algorithm cant depend on the bitstring because lazyEvalHash allows it not to exist yet.
It must depend on the sizeList() number of childs.
All we have from those is their hashInt32 which will be cached by each constructor.
Use a similar randomPerJvm array as in the other algorithm, except int32s
(or maybe both should just be int32s?). Xor each child 1 of those, and sum all that.
Or maybe Zing should have a hash64 func and derive hash32 from that at end?
Map and list are the same datastruct except for 1 bit in the mask,
so at the end, not~ all the int32 bits if map.
<br><br>
"TODO the 4 hashInt32 algorithms"
"TODO should zing have a hashCode64 func"
"No, it should not have hashCode64. It should use int32 in the fast merkle because I dont want to store int64."
"I want to use the same randomPerJvm array for all the hash32 algorithms, so either they must both be int32 or use the int64s as 2 int32s or use only 1 of the int32s or multiply the int32 by the int64 then take its high 32 bits. I like the multiply solution, though it does waste on average about 16 of the higher bits, and its main problem is the speed of multiplying long*int per child"
"I've decide, do it: I'm considering the map and list hash to use the same long array and multiply hashCode()*randomPerJvm[childIndex] and wrappedSum64 those, and at end not~ the 32 bits if its map. My main concern is the speed of long*int per child. But that probably wont be the bottleneck compared to the creation and reading of objects, so do it."
"This isnt well explained. I'm implementing it in Z.hash32(Zing)."
*/
public interface Zing extends ReadPrimAtBitIndex, Comparable{
	
	//TODO redesignZingToHaveBasicTypes
	
	/** Bit index 0. If true, header is in 64 bit blocks, else 16. */
	public default boolean isBig(){
		return ja(0) < 0;
	}
	
	/** Bit index 1. If true, is the map kind of maplist. Hash is a leaf. */
	public default boolean isMap(){
		return (ja(0)&ZingRoot.maskIsBig) != 0L;
	}
	
	/** Bit index 2. If true, is the list kind of maplist. Hash is leaf. */
	public default boolean isList(){
		return (ja(0)&ZingRoot.maskIsList) != 0L;
	}
	
	/** Bit index 3. */
	public default boolean isHash(){
		return (ja(0)&ZingRoot.maskIsHash) != 0L;
	}
	
	/** Is a leaf bitstring. Same as !(isMap() || isList() || isHash()) */
	public default boolean isLeaf(){
		return (ja(0)&(ZingRoot.maskIsMap|ZingRoot.maskIsList|ZingRoot.maskIsHash)) == 0L;
	}
	
	/** same as isMap() || isList() but may be faster */
	public default boolean isMaplist(){
		return (ja(0)&(ZingRoot.maskIsMap|ZingRoot.maskIsList)) != 0L;
	}
	
	/** Bit index 4 to isBig()?63:15.
	Size 0 if leaf else size of the list or twice the size of the map which alternates key val.
	*/
	public default long sizeList(){
		long g = ja(0)&0x0fffffffffffffffL;
		return isBig() ? g : g>>>48;
	}
	
	/** Bit index 4 to isBig()?63:15. Size of the whole zing including header. */ 
	public default long sizeBits(){
		return isBig() ? ja(1) : sa(1)&0xffff;
	}
	
	/** Bit index (isBig()?64:16)*(listIndex+1) to (isBig()?64:16)*(listIndex+2)-1. 
	Same as get(listIndex).sizeBits() but often far more efficient because reads
	this from header instead of creating Zing view or copy of that range.
	*/
	public default long sizeBits(long listIndex){
		//TODO optimize by only checking isBig once, copying code from start(long) and endExcl(long)
		return endExcl(listIndex)-start(listIndex);
	}
	
	/** bit index where list item starts.
	Subclasses that know their header is smaller than 8gB can override this more efficiently using ja(int).
	*/
	public default long start(long listIndex){
		long sizeList = sizeList();
		if(isBig()){
			if(listIndex == sizeList) return (sizeList+1)*64;
			return j((listIndex+2)<<64);
		}else{
			if(listIndex == sizeList) return (sizeList+1)*16;
			return sa(2+listIndex)&0xffff;
		}
	}
	
	/** bit index (exclusive) where list item iends */
	public default long endExcl(long listIndex){
		if(isBig()){
			return j((listIndex+1)<<64);
		}else{
			return sa(1+listIndex)&0xffff;
		}
	}
	
	/*TODO choose get, listGet, mapGet, etc, funcs, considering every list is a map whose keys are longs,
	but also considering map is a list that alternates key val. These are a consistent design but
	the words contradict. Need to name these differently.
	...
	Write detailed names that explain the whole meaning, then choose abbrevs:
	...
	getByListIndexInList
	getByListIndexInMap
	getByMapKey
	*/
	
	/** If isList(), same as mapGet(long) or mapGet(Zing of that long).
	If isMap(), is the same datastruct as list except a header bit and alternates key val,
	so returns key if even and val if odd.
	If out of range, may throw or return invalid data such as array[(int)listIndex].
	*/
	public Zing listGet(long listIndex);
	
	/** Every list is a map whose keys are longs.
	If isList(), same as listGet(key). If isMap(), can be sparse and include any ptr(Zing) as key.
	If not found (including if this is not a maplist so has no keys or values), returns null.
	*/
	public default Zing mapGet(long key){
		throw new Todo("binarySearch using start(long), sizeBits(long) (must be 96 to match), and j(long)");
	}
	
	/** If not found (including if this is not a maplist so has no keys or values), returns null.
	If isList(), has only long keys in range 0 to sizeList()-1.
	If isMap(), keys are even listGet index, and vals are odd.
	List is a maplist whose keys are all long, so if key is a leafBitstring of sizeBits 96,
	thats 64 bits of content, and this must return the same as mapGet(key.j(32)).
	*/
	public default Zing mapGet(Zing key){
		long sizeList = sizeList();
		if(sizeList < 16){
			//Map size is half listSize
			//TODO always binarySearch? I'm just doing this to get it coded fast
			for(long i=0; i<sizeList-1; i+=2){
				int c = key.compareTo(listGet(i));
				if(c == 0) return listGet(i+1);
				//TODO optimize by returning null early if pass it. Which of neg or pos would c be?
			}
			return null;
		}else{
			throw new Todo("binarySearch using listGet(long)");
		}
	}
	
	/** wraps ZingRoot.s(String) which is a Lang that converts between String and Zing.
	FIXME? Account for the object created or choose to ignore it since it will be garbcoled soon.
	*/
	public default Zing mapGet(String key){
		return mapGet(ZingRoot.s(key));
	}
	
	/** REMOVED THIS BECAUSE ECONACYC SHOULD CALL get on all sizeList childs
	and let them be lazy within those views if they are views.
	<br><br>
	True if get(listIndex) would create or lookup a child object but doesnt exist here yet.
	This is a stateful func despite Zing being stateless, a paradox explained by it being
	an optimization to compute econacyc which is deterministic of any forest shape
	but stateful if new parents are added.
	*
	public boolean getLazy(long listIndex);
	*/
	
	public default Iterable<Map.Entry<Zing,Zing>> iterMap(){
		//TODO Is there an existing immutable Map.Entry so I dont have to create an inner class here?
		final Zing z = this;
		return new Iterable<Map.Entry<Zing,Zing>>(){
			public Iterator<Map.Entry<Zing,Zing>> iterator(){
				return new Iterator<Map.Entry<Zing,Zing>>(){
					long nextIndex = 0;
					final long sizeList = z.sizeList();
					public boolean hasNext(){
						return nextIndex < sizeList-1;
					}
					public Map.Entry<Zing,Zing> next(){
						final Zing key = listGet(nextIndex), value = listGet(nextIndex+1);
						nextIndex += 2;
						return new Map.Entry<Zing,Zing>(){
							public Zing getKey(){ return key; }
							public Zing getValue(){ return value; }
							public Zing setValue(Zing value){
								throw new UnsupportedOperationException("zing is immutable");
							}
							public boolean equals(Object o){
								if(!(o instanceof Map.Entry)) return false;
								return key.equals(((Map.Entry)o).getKey())
									&& value.equals(((Map.Entry)o).getValue());
							}
						};
					}
				};
			}
		};
	}
	
	/** Views as list, so if it is a map it will alternate key val key val */
	public default Iterable<Zing> iterList(){
		final Zing z = this;
		return new Iterable<Zing>(){
			public Iterator<Zing> iterator(){
				return new Iterator<Zing>(){
					long nextIndex = 0;
					final long sizeList = z.sizeList();
					public boolean hasNext(){ return nextIndex < sizeList; }
					public Zing next(){ return listGet(nextIndex++); }
				};
			}
		};
	}

	/** self as Iteratable is of Map.Entry<Zing,Zing> if isMap(), of Zing if isList(), else empty *
	public default Iterator<?> iterator(){
		if(isMaplist()){
			return isMap() ? iterMap().iterator() : iterList().iterator();
		}else{
			return Collections.emptyIterator();
		}
	}*/
	
	/** Returns from cache of last call if exists, so the object must be immutable.
	Interprets my bits as java code and returns any java lambda type such as DoubleBinaryOperator,
	(TODO) but only if its made only of parts that each pass the whitelist.
	The lambda func must have no side-effects and normally operates on Zings andOr primitives.
	Could return any type, maybe a string if somehow the implementation of zing knows thats what its used for,
	but in general you cant go wrong by trying for compiling the code of java lambda (of various possible types).
	If it doesnt compile, its still correct to try that. TODO should it throw
	vs return something that throws vs return a Zing message about what didnt compile?
	Calling compile on a hash will try to get the full version and compile that.
	*/
	public Object compile();
	
	/** Hashes this zing by the algorithm named by an IPFS hash byte
	but this software is otherwise not related to IPFS.
	Adds this Zing to the ZingRoot synchronized WeakHashMap system by hash,
	if it doesnt already exist there (and TODO what if it exists as a different hash?),
	then returns the hash. After calling this,
	if not become garbcolable, ZingRoot.get(hash) returns this zing. 
	<br><br>
	A hash must never be bigger than Z.maxChildLiteralSize.
	<br><br>
	At first is a zingLazyHashStub, until observe bits of that hash except its int16 hashAlg,
	which is needed for the WeakReference system.
	In a hash, the short is at bit index isBig()?128:32 to isBig()?143:47,
	and probably no hash will ever be isBig() but just in case (so 32 to 47).
	Returns a hashname from Hash.hasher(hashAlg).apply(this) and caches the last hash returned.
	That means if a hash is cached, check these 16 bits in it, and if they match, return it,
	else replace that cached hash with from a different hashAlg. They can always be rederived.
	<br><br>
	Example: A sha256 is 304 bits (38 bytes):
	4 mask, 12 listSize(is 0), 16 bitSize(of this zing), 16 hashAlg, 256 content.
	<br><br> 
	*/
	public Zing hash(byte hashAlg);
	//public Zing hash(short hashAlg);
	
	/** REMOVED THIS FUNC BECAUSE IT CAUSES ZING TO HAVE STATE VISIBLE TO CALLER.
	If hash(byte) has been called, returns the hash from its last call,
	else calls it on Hash.defaultHashAlgorithm.
	*
	public Zing hash();
	*/
	
	/** If !isHash() returns this.
	If isHash(), returns content that this zing is the hash of.
	Inverse of every hash(byte) (of course excluding possible collisions in weaker hashAlgs).
	<br><br>
	About the default implementation:
	WARNING: slow because uses the synchronized WeakHashMap system.
	TODO content hashed, or should this be done through Z.get(Zing),
	and what if getting it remotely and need params of that func of max wait time,
	computeMoney, memoryMoney, econacyc, etc?
	*
	protected Zing hsah;
	*/
	public default Zing hsah(){
		if(!isHash()) return this;
		Zing z = ZingRoot.get(this);
		if(z == null) throw new Todo("Get it from harddrive or internet with various download options, max risk, and failure conditions, etc.");
		return z;
	}
	
	public default int compareTo(Object o){
		return compareTo((Zing)o);
	}
	
	/** by length, then by content */
	public default int compareTo(Zing z){
		long mySiz = sizeBits(), zSiz = z.sizeBits();
		if(mySiz != zSiz) return mySiz<zSiz ? -1 : 1;
		int cycles = (int)((zSiz+63)/64); //round up to hang off end and get bit0s there in both
		for(int i=0; i<cycles; i++){
			int c = ZingRoot.compareUint64(ja(i), z.ja(i));
			if(c != 0) return c;
		}
		return 0;
	}
	
	/** If sizeBits() is small enough (TODO what standard limit?), returns this,
	else returns hash(hashAlg). 
	*
	public Zing pointer(byte hashAlg);
	*/
	
	/*public default boolean equals(Object o){
		if(!(o instanceof Zing)) return false;
		throw new Todo("Root/WeakReference system == vs equals. Should Zing implement Comparable (instead of using Comparator)?");
	}*/
	
	/** Return a constant,
	estimated bits of memory used by this zing not including pointers to other zings.
	Childs that have their own copy of data are not included.
	TODO what to return for child views into my same array?
	Return bit size of the whole array (since they prevent its garbcol)?
	Return only size of it they use (despite its preventing garbcol of other)?
	Return only size of them as object (despite its preventing garbcol of whole array it points at part of)?
	*/
	public double econacycLocalCost(); 

}
