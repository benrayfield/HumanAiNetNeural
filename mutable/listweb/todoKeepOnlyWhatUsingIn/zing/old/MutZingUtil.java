package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;

public class MutZingUtil{
	private MutZingUtil(){}
	
	public static long readLongAsBits(MutZing z, long bitIndex){
		long g = 0;
		for(int i=0; i<64; i++){
			if(z.z(bitIndex+i)) g++;
			g <<= 1;
		}
		return g;
	}
	
	public static void writeLongAsBits(MutZing z, long bitIndex, long data){
		for(int i=63; i>=0; i--){
			z.z(bitIndex+1, (data&1)!=0);
			data >>>= 1;
		}
	}
	
	public static boolean areAllSameScalar(double[] d){
		for(int i=1; i<d.length; i++){
			if(d[0] != d[i]) return false;
		}
		return true;
	}
	
	/** same params as System.arraycopy
	TODO UPDATE CODE FOR: Theres no ds funcs, and any range is allowed.
	OLD...
	TODO optimize, use System.arraycopy if types are known and internals can be accessed,
	and maybe there should be funcs in Zing to copy that way,
	but only if it doesnt break encapsulation of immutables.
	Its ok for an immutable to see the internals of a mutable
	as long as its trusted to not write outside of fraction range.
	Its probably fast enough to do it as 1 long or double at a time
	since java functions are very low overhead, almost as fast as primitives.
	*/
	public static void copyRange(MutZing from, long fromIndex, MutZing to, long toIndex, long size){
		if(!from.preferScalarInsteadBit() && !to.preferScalarInsteadBit()){
			throw new Todo("optimize by copying longs except the last up to 63 bits");
		}else{
			//copy scalars (which may be any range)
			for(long i=0; i<size; i++){
				to.d(toIndex+i, from.d(fromIndex+i));
			}
			/*if(from.preferFractionInsteadInverseSigmoid() || to.preferFractionInsteadInverseSigmoid()){
				//If either prefers fraction, copy as fraction
				for(long i=0; i<size; i++){
					to.d(toIndex+i, from.d(fromIndex+i));
				}
			}else{
				//If either prefers fraction, copy as fraction
				for(long i=0; i<size; i++){
					to.ds(toIndex+i, from.ds(fromIndex+i));
				}
			}*/
		}
	}
	
	/** Returns immutable only if both are immutable.
	The 2 params remain garbcolable because its copied, not just pointing at them.
	*/
	public static MutZing copyConcat(MutZing x, MutZing y){
		throw new Todo();
	}
	
	/** Returns a backed view of the concat of those 2 */
	public static MutZing wrapConcat(MutZing x, MutZing y){
		throw new Todo();
	}
	
	public static void fill(MutZing z, double value){
		for(long i=0; i<z.size(); i++) z.d(i, value);
	}

}