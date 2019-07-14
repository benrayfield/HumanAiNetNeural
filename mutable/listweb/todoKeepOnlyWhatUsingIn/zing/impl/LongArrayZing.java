package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.lg;

import java.util.*;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.*;

/** As usual in zing, size is in bits, not in units of array indexs
<br><br>
Uses long array instead of int array because more computers will be 64 bit
which can read and write a long at once.
TODO
Zings loaded from bitstring, FORK[
	if this zing is bigger than zingMaxChildLiteralSize, then lazyCopy direct childs,
	and that child should create child views into same long array of max bits zingMaxChildLiteralSize.
	UPDATE: wrote that wrong. lazyCopy means not view.
]ELSE[
	this zing is at most zingMaxChildLiteralSize, and all childs should be views of my array.
]
Zings created from existing zings as childs (which are each of course max zingMaxChildLiteralSize, as literal or hash),
FORK[
	if this zing is bigger than zingMaxChildLiteralSize therefore wont be included as literal in any zing,
	then instantly copy all childs bits to new long array.
	UPDATE: lazyCopy all my bits the first time any of them are read.
	UPDATE: Wait, if its at most zingMaxChildLiteralSize, dont copy,
	since must hash if its as deep as list(list(x,y),z). 
]ELSE[
	parent zing is at most zingMaxChildLiteralSize so could be included as literal in other zings,
	then copy to my new long array (worst case 20 times the data, average maybe 2.5 times the data),
	because the alternative is that much branching for each primitive read, and most ops are reads,
	but the system will still be very efficient because most of the data
	is large arrays like of float32 for neuralnets, and thats not duplicated.
]
UPDATE: Also require hash of any child bigger than zingMaxChildLiteralSize or deeper than 1,
so list(sha256,sha256) and list('txt',sha512) are both ok,
and list(list(x,y),z) would hash list(x,y) and use z as literal.
So now the worst case for duplication is much lower.
<br><br>
Rewriting these rules...
<br><br>
boolean doHash = zingMaxChildLiteralSize < sizeBits() || ???theUpdateForDeeperThan1???;
if(doHash){
	if(constructUsingBitstring){
		lazyCopy from my array to new child when call get and find that index of array is null.
		FIXME depending on ???theUpdateForDeeperThan1??? beware recursive copying/recursion?
	}else{ //construct using existing zings as childs
		lazyCopy from child to to my bits when read the first of my bits thats not part of my header.
		FIXME depending on ???theUpdateForDeeperThan1??? beware recursive copying/recursion?
	}
}else{ //at most size zingMaxChildLiteralSize and (TODO?) shallow enough
	if(constructUsingBitstring){
		lazyView childs into my same array (null until then),
		so only the child objects recursed into will be created,
		and none of those will copy the data, just the overhead of an object.
		This can prevent at most constructUsingBitstring plus those objects from being garbcoled.
		FIXME depending on ???theUpdateForDeeperThan1??? beware recursive copying/recursion?
	}else{ //construct using existing zings as childs
		TODO depending on ???theUpdateForDeeperThan1???, should this be instant copy vs lazyViewR?
		FIXME depending on ???theUpdateForDeeperThan1??? beware recursive copying/recursion?
	}
}
Wait...
What bout that "UPDATE" about "deeper than 1"?
FIXME depending on ???theUpdateForDeeperThan1??? beware recursive copying/recursion?
<br><br><br>
doHash means to choose to include a child as hash vs literal.
Its ok not to choose the copy vs recurse strategy yet, since the only thing that affects the hashes
is the recogFunc(Zing) which tells if that zing must be used as literal vs hashed, in Z.ptr(Zing).
Thats the next step. The other parts are optimization.
A deterministic calculation of doHash is needed for dedup.
Dedup reduces wasted memory. Dedup is the only way to on different computers derive shared names
of objects and have it be the same name if derived as the same resulting data.
<br><br>
boolean doHash = zingMaxChildLiteralSize < sizeBits() || ???theUpdateForDeeperThan1???;
boolean doHash = WHAT?
<br><br>
If zingMaxChildLiteralSize < sizeBits(), certainly doHash.
<br><br>
The big question is, how should localDepth (not past hashes) affect doHash?
<br><br>
If it doesnt affect, then localDepth can be up to 19 (lists of size 1 with empty string at end),
but estimate average localDepth at 2.5 where not trying to slow the system and waste memory.
econacyc could handle that by defining accurate localCost per zing, so lots of small objects
cost more than few big objects of the same bit size, of course spreading cost for duplicates,
and cost can mean any kind of constant cost including memory and average computing time.
<br><br>
Any data that intentionally wastes computing resources will be penalized for it
in which data is likely to spread and chance of keeping network connections.
The network will be driven toward occamsRazor strategies.
<br><br>
It still bothers me that gametheory would be needed to handle such a basic thing
that could be solved with a hard rule, since gametheory is expensive.
<br><br>
On the other hand, the hard rule ???theUpdateForDeeperThan1??? could create more size
by creating up to 19 (or half of that?) hashes
instead of including the up to zingMaxChildLiteralSize literally.
<br><br>
Also consider that hashes are expensive, and its better to have less of them and more literals.
Hashes are expensive because of the time to compute them and the synchronized WeakHashMap system.
<br><br>
The choice FOR vs AGAINST theUpdateForDeeperThan1, for zings at most zingMaxChildLiteralSize,
is only relevant when constructing zing from existing childs
because when constructing from bitstring it will lazyView into the same array.
<br><br>
FOR theUpdateForDeeperThan1:
* localDepth at most 2 (such as list(...,list(type,value),...)).
* Always fast enough to use recursion to get bits starting from a child (but still lazyCopy childs to my bits if read my bits). 
<br><br>
AGAINST theUpdateForDeeperThan1 (and FOR more recursion andOr copying):
* less hashes, instead smaller literals.
* Fast enough to use for recursion if not trying to make it slow by using unnecessarily deep recursion. 
<br><br>
Should theUpdateForDeeperThan1 only apply to size 2 list (thats at most zingMaxChildLiteralSize bits)?
I designed this optimization for list(type,value), but should it be more general?
I choose only size 2 list because it can be more optimized by using a constant var of size of first child
instead of searching an array of sizes.
<br><br>
Whatever the rule is for computing doHash, it will be computed in Z.ptr(Zing). 
TODO
*/
public class LongArrayZing implements Zing{
	
	static{ lg("TODO the logic in comment of this class, and decide on that logic before build it"); }
	
	/** all sizes are in bits *
	public final long size;
	*/
	
	/** TODO should this be created as lazyEval to avoid copying childs bits here
	unless its called enough to need that optimization?
	The optimization is randomAccess instead of the log cost of binarySearch,
	considering that it could do multiple binarySearches recursively into childs
	and in the worst case do linear number of binarySearches if the childs are each
	nearly the size of the parent, so its often better to copy that.
	*/
	protected final long[] data;
	
	/** For the WeakReference system, must keep these zingLazyHashStubs and small literals */
	protected final Zing child[];
	
	/** see comment of Zing interface and Z.hash32(Zing) for exactly what this must be,
	for compatibility with Comparable and equals.
	*/
	public final int hash32;
	
	protected Zing hash;
	
	/** bit size and other properties are in the data. Bit size can use 1-64 bits of the last long.
	WARNING: Uses the array directly, and Zing is immutable, so it must never be modified again.
	*/
	public LongArrayZing(long[] data){
		this.data = data;
		long sizeBits = sizeBits();
		long arrayBits = data.length<<6;
		if(sizeBits < arrayBits-63 || arrayBits < sizeBits) throw new Err(
			"sizeBits="+sizeBits+" arrayBits="+arrayBits+" but must use 1-64 bits of last long.");
		this.child = newChildArray(); //FIXME lazyCopy each child when first get
		hash32 = ZingRoot.hash32(this);
	}
	
	/** copies the array *
	public LongArrayZing(long[] data, long size){
		this.data = data.clone();
		this.size = size;
		this.child = newChildArray();
		hash32 = ZingRoot.hash32(this);
	}*/
	
	static{lg("TODO should LongArrayZing(boolean,Zing...) constructor change map to list map can be normed to a list, or should it throw if its not? Must norm.");}

	public LongArrayZing(boolean wantMap, Zing... childs){
		this.child = childs.clone();
		//FIXME? For now I'm copying the childs instantly, but TODO? lazyCopy if this.isBig() else j(long) recurses into childs
		//FIXME verify keys (even indexs) are sorted ascending, and
		//maybe I should for now copy to TreeMap and call the same code as in LongArrayZing(Map<Zing,Zing> map)?
		if(wantMap){
			//TODO merge duplicate code from LongArrayZing(Map)
			/*int c = 0;
			boolean isMap = false;
			for(int c=0; c<childs.length; c++){
				
			}
			for(Map.Entry<Zing,Zing> entry : new TreeMap<Zing,Zing>(map).entrySet()){
				child[c] = entry.getKey();
				if(child[c].sizeBits() != 96 || child[c].j(32) != c/2){
					//Norm: its a List if all keys are long (32+64 bits each) and include 0L to map.size()-1L. This norm is required.
					isMap = true;
				}
				child[c+1] = entry.getValue();
				c += 2;
			}
			data = ZingRoot.longArrayFromHeaderAndCopyChilds(isMap, child);
			*/
			throw new Todo();
		}else{
			data = ZingRoot.longArrayFromHeaderAndCopyChilds(wantMap, child);
			hash32 = ZingRoot.hash32(this);
		}
	}
	
	/** TODO should Lang do this instead? */
	public LongArrayZing(Map<Zing,Zing> map){
		//TODO merge duplicate code from LongArrayZing(boolean,Zing[]) where boolean is true for wantMap
		child = new Zing[map.size()*2];
		int c = 0;
		boolean isMap = false;
		for(Map.Entry<Zing,Zing> entry : new TreeMap<Zing,Zing>(map).entrySet()){
			//Dont have to wrap TreeMap in Collections.unmodifiableNavigableMap if dont return it
			child[c] = entry.getKey();
			if(child[c].sizeBits() != 96 || child[c].j(32) != c/2){
				//Norm: its a List if all keys are long (32+64 bits each) and include 0L to map.size()-1L. This norm is required.
				isMap = true;
			}
			child[c+1] = entry.getValue();
			c += 2;
		}
		data = ZingRoot.longArrayFromHeaderAndCopyChilds(isMap, child);
		hash32 = ZingRoot.hash32(this);
	}
	
	/** TODO should Lang do this instead? */
	public LongArrayZing(List<Zing> list){
		this(false, list.toArray(new Zing[0]));
	}
	
	
	/** For the WeakReference system, must keep these zingLazyHashStubs and small literals */
	protected Zing[] newChildArray(){
		long sizeList = sizeList();
		if((1<<29) <= sizeList){
			throw new Err("While "+Zing.class.getName()+" supports as many childs as fit in 2^64-1 bits,"
				+" this kind only supports up to 2^29-1 childs. sizeList="+sizeList);
		}
		Zing childs[] = new Zing[(int)sizeList];
		if(childs.length != 0){
			//Copy childs as subrange of my bits.
			//FIXME? should Root class (WeakReference system) be checked for the normed form of them if they're hashes?
			//FIXME? If root class is not checked for dup, then how should the child array be updated?
			//FIXME? How does this work with lazy-eval-hash?
			boolean big = isBig();
			for(int i=0; i<childs.length; i++){
				long endExcl = big ? ja(1+i) : sa(1+i*2)&0xffff;
				/*long start = 
					? (big ? ja(2+i) : sa(2+i*2)&0xffff)
					: (big ? ja(2+i) : sa(2+i*2)&0xffff);
				*/
				throw new Todo();
			}
			throw new Todo();
		}
		return childs;
	}
	
	/** Inefficient because copies to long array, unlike the other constructor which uses the array directly */
	public LongArrayZing(byte[] bytes){
		this(ZingRoot.toLongArrayFromByteArray(bytes));
	}
	
	/** copies the array *
	public LongArrayZing(byte[] bytes, long size){
		data = new long[(int)((size+63)>>>64)];
		for(int i=0; i<data.length; i++){
			data[i] = ZingRoot.longInByteArrayByteAlignedCanHangOffEnds(bytes, i<<3);
		}
		this.size = size;
		this.child = newChildArray(); //FIXME lazyCopy or lazyView
		hash32 = ZingRoot.hash32(this);
	}
	
	/** copies the array *
	public LongArrayZing(byte[] data){
		this(data, data.length*8L);
	}
	*/

	public long j(long bitIndex){
		int i = (int)(bitIndex>>>6);
		long highBits = data[i];
		int remain = (int)(bitIndex&63);
		//if(remain == 0) return highBits;
		long lowBits = data.length<=i+1 ? 0 : data[i+1];
		return (highBits<<remain) | (lowBits>>>(64-remain));
	}
	
	public long ja(long longIndex){
		return data[(int)longIndex];
	}
	
	public Zing listGet(long listIndex){
		//FIXME? could wrap into valid range, but expecting IndexOutOfBoundsException, but dont want to pay to check for that
		return child[(int)listIndex];
	}
	
	/** if zing, by bitstring */
	public boolean equals(Object o){
		if(this == o) return true;
		if(!(o instanceof Zing)) return false;
		return compareTo((Zing)o) == 0;
		//throw new Todo("Root/WeakReference system == vs equals. Should Zing implement Comparable (instead of using Comparator)?");
	}
	
	public int hashCode(){ return hash32; }

	public Object compile(){
		throw new Todo();
	}

	public Zing hash(byte hashAlg){
		//while technically allowed, a hash will never be isBig() which is over about 8kB
		if(hash != null && hash.ba(4) == hashAlg) return hash;
		hash = ZingRoot.hasher(hashAlg).apply(this);
		lg("FIXME should only ZingRoot.put when LazyEvalHash actually does the hash, and make sure any trying to compare a lazyHashStubs (against another lazyHashStub or any zing) causes the hash and puts it in ZingRoot's WeakHashMap system.");
		ZingRoot.put(this);
		return hash;
	}
	
	public Zing hsah(){
		Zing z = ZingRoot.get(this);
		//FIXME if(z == null) throw new Todo("Get it from harddrive or internet with various download options, max risk, and failure conditions, etc.");
		return z;
	}

	public double econacycLocalCost(){
		throw new Todo();
	}
	
	public String toString(){
		if(isMap()) return "[map "+sizeList()+"/2 bits="+sizeBits()+"]";
		if(isList()) return "[list "+sizeList()+" bits="+sizeBits()+"]";
		if(isHash()) return "[hash (TODO hex content)]";
		return "[leaf bits="+sizeBits()+" (TODO hex andOr string content)]";
	}

}