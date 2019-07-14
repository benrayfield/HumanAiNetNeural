package mutable.listweb.todoKeepOnlyWhatUsingIn.zing;

/** Context. Anything stateful. All stateless parts must be done through zings andOr primitives.
All stateful parts must done through this to keep them sandboxed.
I'm planning computeMoney and memoryMoney of econacyc,
which means all allocation of objects goes through here,
which makes this a factory, among other uses.
TODO add those funcs.
*/
public interface Cx{
	
	/** memoryMoney can be reused. */
	public long memFree();
	
	/** For zings not to be garbcoled, they must have enough memoryMoney from incomingPointers to cover that cost. */
	public long memUsing();
	
	/** same as memFree()+memUsing() */
	public long memTotal();
	
	/** computeMoney is consumed when used. */
	public long compFree();
	
	/** uses this much computeMoney if exists here then returns true, else false */
	public boolean compUse(long use);
	
	/** Gets a Zing by hash if hash(byte) has been called on it and that hash (by equals)
	has not been garbcoled since last created. Theres no public function in this interface for
	the counterpart "put" because its done automaticly by hash.
	*/
	public Zing get(Zing hash);
	
	/** See ZingRoot.ptr(Zing) */
	public Zing ptr(Zing z);
	
	/** returns the empty list which is the normed form of an empty maplist.
	maplist.isMap() && maplist.sizeList()==0 is an error.
	*/
	public Zing emptyMaplist();
	
	/** Returns the empty leafBitstring */
	public Zing emptyLeaf();
	
	/** reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow?
	Since every list is abstractly a map whose keys are longs, all map ops work on lists.
	*/
	public Zing mapPutAll(Zing firstMaplist, Zing secondMaplist);
	
	/** reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow?
	Since every list is abstractly a map whose keys are longs, all map ops work on lists.
	*/
	public Zing mapPut(Zing maplist, Zing key, Zing val);
	
	/** reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow? */
	public Zing listCat(Zing firstList, Zing secondList);
	
	/** concats leaf bitstrings and returns a new leaf bitstring. The sizeBits() is near the sum except for having 1 less header.
	Reduces memoryMoney until thats garbcoled, but TODO how to do that without finalize() which is very slow?
	*/
	public Zing leafCat(Zing firstLeaf, Zing secondLeaf);
	
	/** func is either a leaf whose content is a string such as "java:...then some java code" or a hash of that.
	Reduces*/
	public Zing funcall(Zing func, Zing param);
	
	public default Zing funcall(Zing func, String param){
		return funcall(func, ZingRoot.s(param));
	}
	
	/** True if using the econacyc math (normally cached because its impractically slow to do it exactly)
	If econacyc, garbcol zings which have 0 econacycInPtrs(Zing),
	in descending order of econacycRelValue(Zing)/econacycCost(Zing,recentTime).
	EconacycRelValue is normally how much memoryMoney is stored in that Zing
	but could be a function of proofOfWork or chosen arbitrarily
	or put only on the ZingRoot.threadState.get() current root zings 1 per thread. 
	*/
	public boolean econacyc();
	
	/** Time is the farthest back allow use cached econacycCost recursively.
	Updates that cache to time if its newer, else returns from cache.
	This equals (econacycLocalCost(aZing)+sum(econacycCost(eachChild,recentTime)))/econacycInPtrs(aZing),
	where eachChild is the Zing.get(long) from 0 to Zing.sizeList()-1.
	*/ 
	public double econacycCost(Zing z, double time);
	
	public double econacycInPtrs(Zing z);
	
	public double econacycOutPtrs(Zing z);
	
	public double econacycRelValue(Zing z);

}
