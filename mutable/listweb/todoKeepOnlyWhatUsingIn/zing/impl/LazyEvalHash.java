package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.*;

/** a lazy-eval-hash of another zing. Knows its size and byte hashAlg before triggering lazy-eval,
but not any other bits. When trigger, hashes the other zing, which may trigger recursive
lazy-evals in its forest of childs
(TODO verify once lazy-eval starts it wont threaded start again until it finishes),
and then all bits of the hash are known.
*/
public class LazyEvalHash implements Zing{
	
	public final Zing target;
	
	/** The first 40 bits of a hash are known before computing the hash.
	They are:
	4 bits of mask. The last of these is the isHash bit.
	12 of listSize(0).
	16 of bitSize (get from Hash.sizeOfHash(byte)).
	8 hashAlg.
	These are the high 40 bits of the long.
	Then hash.
	<br><br>
	Dont preallocate the array for the whole hash since most lazy-eval-hash are never triggered,
	as its a general computing system, and hashes are only for storage, networking, and dedup.
	*/
	protected final long first40Bits;
	
	/** TODO more efficient to do this as long[] but for now do it the easy way */
	protected Zing fullHash;
	
	public LazyEvalHash(Zing target, byte hashAlg){
		this.target = target;
		//1<<60 isHash bit
		first40Bits = (1L<<60) | ((ZingRoot.hashSize(hashAlg)&0xffffL)<<32) | ((hashAlg&0xffL)<<24);
	}

	public long j(long bitIndex){
		if(fullHash != null){
			return fullHash.j(bitIndex);
		}else{
			throw new Todo();
		}
	}

	public int compareTo(Zing o){
		if(fullHash != null){
			return fullHash.compareTo(o);
		}else{
			throw new Todo();
		}
	}
	
	public Zing listGet(long listIndex) throws IndexOutOfBoundsException{
		if(fullHash != null){
			return fullHash.listGet(listIndex);
		}else{
			throw new Todo();
		}
	}
	
	public Zing mapGet(long key){
		return null; //hash has no childs
	}
	
	public Zing mapGet(Zing key){
		return null; //hash has no childs
	}

	public Object compile(){
		if(fullHash != null){
			return fullHash.compile();
		}else{
			throw new Todo();
		}
	}

	public Zing hash(byte hashAlg){
		throw new UnsupportedOperationException("Already a hash");
	}

	public double econacycLocalCost(){
		throw new Todo();
	}

}