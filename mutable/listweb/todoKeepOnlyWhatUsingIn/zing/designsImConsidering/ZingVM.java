package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.designsImConsidering;

import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Zing;

/** Zing Virtual Machine. Econacyc.
All meta-garbcol and allocation of objects and computing cycles by code in zing data
should be done through this interface. UI and access to graphics and sound is outside this scope
which is only about zings and primitives.
Each instance of this interface has some amount of memoryMoney and computeMoney
and can be split or merged.
This is not secure money and is for calculations within a single computer
to prevent AIs and other untrusted code from infinite looping or other wastes.
The global p2p network, if it has any memoryMoney and computeMoney layer,
would have to use a different system than this interface.
<br><br>
*/
public interface ZingVM{
	
	/** memoryMoney can be reused. */
	public long memFree();
	
	/** For zings not to be garbcoled, they must have enough memoryMoney from incomingPointers to cover that cost. */
	public long memUsing();
	
	/** same as memFree()+memUsing() */
	public long memTotal();

	/*
	"ECONACYC: TODO ZingVM (or maybe separate interface called Mem) should be constant amount of memoryMoney balancing the cost of 0 or more zings or maybe just up to 1 specific zing. When the econacycCost of this memoryMoney object exceeds its long value, its returned to its creator which is some system of summing or organizing memoryMoney (maybe another memoryMoney object?). In that way, a memoryMoney object is no different than any other object in the econacyc forest."
	...
	"Every few seconds (or whatever small interval), all zings would have to update their approximate cached econacycCost, in order of globalHeight, to the sum of cachedEconacycCost/inPtrs of all their childs plus localCost, but..."
	"...but since that could get expensive, I've removed the dependency on econacyc and instead garbcol whatever is not reachable from the current Zings at ZingRoot.threadState 1 current Zing per thread, and to decide what is the next state a func may optionally compute econacyc or any other way, so garbcol will be handled by ZingRoot's synchronized WeakHashMap and LazyEvalHash's pointers etc. Garbcol will be handled by java, and deciding what to allow to become javaGarbcolable may be done in any way such as econacyc or arbitrary choice of new state of a thread. Keep in mind that many zings including very large zings (wrapping frozen files for example) will be on harddrive andOr internet and named by hash, so realtime econacyc would be impractical on that scale, so just do it per computer this way instead."
	*/
	
	/** computeMoney is consumed when used. */
	public long c();
	
	/** uses this much computeMoney if exists here then returns true, else false */
	public boolean c(long use);
	
	public ZingVM take(long memoryMoney, long computeMoney);
	
	public Zing empty();
	
	/** reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow? */
	public Zing mapPutAll(Zing firstMap, Zing secondMap);
	
	/** reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow? */
	public Zing listCat(Zing firstList, Zing secondList);
	
	/** concats leaf bitstrings and returns a new leaf bitstring. The sizeBits() is near the sum except for having 1 less header.
	Reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow?
	*/
	public Zing leafCat(Zing firstLeaf, Zing secondLeaf);
	
	/** func is either a leaf whose content is a string such as "java:...then some java code" or a hash of that.
	Reduces*/
	public Zing funcall(Zing func, Zing param);

}
