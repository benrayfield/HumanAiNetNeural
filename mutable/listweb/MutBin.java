package mutable.listweb;

import java.nio.Buffer;
import java.nio.MappedByteBuffer;
import java.sql.Types;
//import org.lwjgl.util.mapped.Pointer;

/** mutable binary space, cached in memory so touches harddrive less often.
Harddrive has its own cache but this is to be used at the speed of memory in this OS process
but I only want it to touch harddrive maybe once per minute, like listweb does for json.
The first use of this will be a .binary file parallel to the .json and
.jsonperline (TODO rename to .jsonl) in listweb, which is not versioned and is
opened either just to read it or as RandomAccessFile. That will be encapsulated
in ListwebRoot static funcs to read and write certain byte range of a mindmap name's MutBin,
which will create the MutBin and file if doesnt exist.
Or maybe I should use nio buffer which is more general?
Can it wrap RandomAccessFile? I know it can do an array just in memory.
*/
public class MutBin{

	/*2019-3-3 redesignOccamsfuncerAsFuncParamReturnWithoutJavaclassPerType QUOTE
	CANCEL THIS CLASS. I WANT STORAGE AND GARBCOL FOR OCCAMSFUNCER,
	WHERE EACH LISTWEB NAME CAN HAVE 1 NONVERSIONED FUNCER. I WONT VERSION THEM CUZ THEY TAKE ALOT OF SPACE,
	BUT I COULD EASILY SINCE THEY'RE IMMUTABLE, OR AT LEAST TO VERSION THE ID USING A CORETYPE.WEAKREF,
	MAYBE MAKE THEM ALL WEAKREFS AND VERSION THEM, EXCEPT THE NEWEST FUNCER IN EACH NAME IS A STRONGREF.
	BUT I DONT WANT TO SAVE THE JSON JUST CUZ ITS FUNCER CHANGES, SO NO VERSIONING ON THE FUNCERS,
	BUT THEY WILL STILL SHARE CHILDS.
	I WANT THE STATE OF THE OCCAMSFUNCERS TO BE IN A FUNCER MAP,
	AND AN EXTERNAL SYSTEM CAN ALIGN THAT TO LISTWEB NAMES, SO OCCAMSFUNCER WILL HAVE JUST 1 STATE AT A TIME.
	The first use of this will be to stream append (by lazycatPairs with avllike balancing) mouse positions
	using MouseRecorder.java. There will be a Var<Funcer> for memory and a Var<Funcer> for harddrive,
	or maybe I'll have a more complex system for saving only some of the deep childs depending
	how long they've been in memory.
	Also I want ability to have a total size of funcers thats much bigger than can fit in memory
	and to load/save on harddrive (maybe using MappedByteBuffer),
	and ability to slowly update fullEconacyc (and maybe memlock some parts into memory) including parts not in memory.
	Maybe use long as localId to save memory, and a table of long to 192bit Id. If its negative, then its a literal,
	such as most of the doubles can be represented (else must put it in id form), all the floats, etc.
	There will be a separate MappedByteBuffer for each CoreType so each can be a regular grid,
	except for array wrappers which are diff sizes of array. Store height as a way to efficiently prove theres no cycles.
	An object is automatically deleted when it has 0 incoming pointers, but it doesnt cause other objects to move
	in the MappedByteBuffer. Objects never move in the MappedByteBuffer. Instead, keep a tree of ranges of memory
	that are free and use the lowest one available.
	Maybe I should only store powOf2 sizes of array, and use log number of them concatted,
	so its still a regular grid in each MappedByteBuffer, such as a MappedByteBuffer of 128 (or should it be 64?) or less bits,
	and another MappedByteBuffer of 1024 bit blocks, etc. That way, never have to move anything.
	Or maybe just dont use the whole size and have more sizes such as 256, 300, 400, 512, 700, etc. Regular grid is important.
	But what if theres a size range of array, such as 13k-15k, that there were alot of, so the MappedByteBuffer is enlarged,
	but later most of them are deleted, but still have some near the end after a bunch of unused space,
	and since I decided not to move anything ever, cant shrink it?
	Maybe it would be better to have just 2 sizes of array pieces: one of them very small, and the other maybe 256 bytes,
	and make the arrays of pieces like that which would tend to be allocated near eachother in whatever free spaces there are
	but still it could get very fragmented.
	Maybe the arrays should be movable (lookup current address by long) but nothing else is movable?
	How about powOf2 size blocks whose address is a long whose high 6 bits are how big the range is (or simply binheap addressing?)?
	Since nonarrays have constant address in MappedByteBuffer (and maybe 20 total MappedByteBuffers for different CoreTypes),
	store those locations and which MappedByteBuffer (and maybe bit size in storage?), which I'll call storageId,
	with each Pointer (or if pointers are abbreved as long then lookup with that long).
	Ids have a magic16 to specify which coreType (which is the high 16 bits of sha256 of the enum name of that CoreType),
	so maybe 16 bits storageId should be that.
	I need ability to lookup object content (what hashes to its id) by id (where that id doesnt have an attached storageId).
	I want a table for all the mutable parts of objects, including storageId, econacycCost, number of incoming pointers.
	I want the MappedByteBuffer always completely filled with objects,
	including EmptySpace objects, to simplify finding the empty space.
	Nonarrays will never move cuz I dont want the complexity of finding reverse pointers.
	Arrays can move. So arrays and nonarrays will go in separate MappedByteBuffers.
	I might put all the nonarrays (multiple coretypes) in the same MappedByteBuffer so dont have to resize things.
	All the coretypes except arrays and num are about the same size, maybe varying from 30-100 bytes. Maybe its ok to use blocks of such maxsize?
	But if mappairs have minkey and maxkey instead of deriving that, and if maps and lazycatpairs have size instead of deriving that,
	deriving such as storing in CacheFuncParamReturn, then it might be wasteful to use the same blocksize for all of them.
	But if I'm storing pointers as longs (which can find the id in the mutableparts table), maybe thats not alot of wasted space.
	If I want to recursively load a funcer from MappedByteBuffer, it should be done in parallel
	by decreasing height starting at the height of the node being loaded (and its childs recursively below),
	so a harddrive read head will only have to move across the harddrive height times, since we dont know
	which of them to load next until we get the next.
	...
	This is getting too complex. Use a java db (such as derby or hsqldb), but make sure it supports at least a terabyte and runs
	in the same OS process as the java. Put all the binary blobs there and other object Types.
	Only use the db for storage, continuously depending if an object is in memory for long enough it gets saved,
	but not for objects that are only in memory which is most objects. Use long primaryKeys (nanoseconds since 1970).
	Load objects with parallel queries from a chosen set that you want all of loaded conditional on
	if some parts are huge below stop loading that branch until ask again for it.
	BUY A SOLIDSTATEHARDDRIVE (usb3) for low lag randomaccess of smaller parts, and run the db on that,
	but run anywhere just recommend such a harddrive.
	...
	Consider using just 2 tables: [func param return] table and blob table.
	Example: [econacycCostAtTime56745645.4345 someObject 445.3] and [minKey someObject itsMinKey] and [left someObject itsLeft].
	A mappair_leaf_branch would need at least 5 of those for minKey minChild maxChild maxKey and size,
	so it would be 5 rows of 3 longs each, so 120 bytes.
	Keep in mind that [func param return] can have cycles such as 2 funcs which when called on "abc" each return the other func,
	or a func that returns itself nomatter its param.
	I'm ok with that inefficiency, at least for now its a way to get started fast,
	IF I can decide on a deterministic way to choose which of those [func param return] to garbcol
	when given any object to keep only those reachable below it.
	ForExample, mappairs certainly need their minChild and maxChild, and it would be good to keep cache of
	their minKey and maxKey and size, and probably want to keep econacycCost of it,
	but just cuz something is cached in memory, which every [func param return] is cached between consecutive video frames,
	does not mean it should be saved in db.
	I want compositekeys of all combos of the 3 things, like func, param, return, funcAndParam, paramAndReturn, funcAndReturn, funcAndParamAndReturn,
	all indexed in realtime. Maybe just saying all 3 together are the primarykey would do that? But does that allow efficient searching
	by just 1 or just 2 of them? If not, can derive other search columns for that though would be wasteful.
	Its already going to be slow from the CacheFuncParamReturn. Maybe should just represent everything as [func param return] triples,
	in memory and in db the same way. The speed comes from some parts optimized as opencl and some parts as acyclicflow,
	but what I want otherwise is mostly gluecode that needs to run much faster than an esolang such as a million ops per second.
	Optimizations can be compiled to double-switch, switch(int) the high bits and inside that switch(int) the low bits,
	so can compile optimizations for any combos of long, by default just the Opcodes which are [? "plus"] etc.
	Define some high magnitude range of longs as ids and use the rest for literals such as float, some doubles, longs, etc.
	...
	Create a new kind of hashtable using buckets whose state is each an immutablelinkedlist of [func param return]
	hashed by [func param], so buckets are usually 0-2 deep (sometimes 3 less often 4 and so on).
	If needed, also copy it as hashed by func but I'm unsure if I'd need that.
	Use maybe 2 instances of this hashtable, one for things that the 192bitids depend on and another for the cache that
	is replaced between each 2 videoframes.
	...
	If a long starts with 4 certain bits (see mindmap I wrote about it, which means its in some high magnitude range
	but is not infinite or nan), then its used as a double, else its used as an id. Or better, use those double values
	as ids so dont have to convert between long and double so is much faster. This is of course the slow interpreted part
	and will still have opencl and acyclicflow optimizations so is a numbercrunching system.
	So localids are doubles in a certain high range that leaves 60 bits for id.
	OR, maybe longs are better cuz still need their bits to compute an int hash for the "new kind of hashtable".
	...
	so theres no javaclasses such as Num and MapPair* and LazycatPair*, but all of those things in CoreType and Opcode enums
	will still exist, just represented differently, considering that every function call was going to be in CacheFuncParamReturn anyways.
	Theres only [func param return] (one hashtable that depends on ids and one thats cleared every video frame)
	and blobs (of type double[] float[] int[] utf8byte[] longAsBit[]) and ids.
	double-switch will be the eval func, still using Wallet.wallet and Wallet.have static fields.
	UNQUOTE.*/
	
	//protected RandomAccessFile file;
	
	//public MutBin(File f){
	//}
	
	/** If true, its just in memory and is lost when program closes.
	If false, it will still be cached for fast use in memory but will forExample every 1 minute write to file,
	and on first read of a range it will read from that file.
	<br><br>
	TODO I might also want temp vars in listweb, but a perm var in listweb can still have a temp MutBin.
	Also I might want a level between temp and nontemp, where it has an expireTime? And maybe a dependnet?
	*/
	protected boolean isTemp;
	
	protected final Buffer buf;
	
	public MutBin(Buffer buf){
		this.buf = buf;
	}

}
