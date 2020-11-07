package immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** childs are: returnArraySize, kernel read param 0, kernel read param1..., ndRange[0], ndRange[1]...
Kernel starts with 1 array to write-only into, then the rest of params are read-only.
<br><br>
This is a little like TensorFlow except much simpler and low level and only within memory
of 1 computer. To flow tensors across many computers, an external caller would have to
make multiple calls of this and fit them together.
Its a layer that libraries such as a new fork of TensorFlow or OccamsFuncer could call,
though there is already an opencl fork of tensorflow thats said to still be buggy and incomplete
but its basic parts working as of Y2019. This is designed for low lag such as in realtime games.
<br><br>
This is a forest of stateless ops such as opencl ndrange kernel calls that each return 1 array,
or such as java code whose behaviors depend only on array params, and does not modify arrays,
which may be of any primitive type, and a get_global_id(0) chooses
which block of n indexs to return in that size howManyGlobalIds*n array.
For example, in multiplying bigints that are size 5 ints each,
n would be 5 (or result is size n=10, inputs are size 5).
For example, in matrix multiply of float, n would be 1
and would each return the dotProduct 2 vectors.
<br><br>
Allocates lwjgl CLMem object per OpenclForestOp and may reuse the if the same size
and not needed, though in early implementation of this it will just alloc and free them every time.
*/
public class ForestOp<T> extends ParallelOp{ //TODO move to package immutable.compilers.forest
	
	/*
	//Mem makes CLMem and 1d primitive java arrays etc interchangible.
	Mem{
		Class elementType; //such as float.class
		int size; //in units of elementType such as byte[size] or float[size]
	}
	//Lock extends ForestOp{
	//	Mem mem;
	//	List<ForestOp> computeChilds; //childs used by nonsandboxedLangColonCode
	//}
	ForestOp{
		Mem mem;
	}
	Write extends ForestOp{
		String nonsandboxedLangColonCode;
		List<Read> childs; //write lock after reads finished
		constraint{ forEach Read r in this.childs, r.mem != this.mem }
	}
	Read extends ForestOp{
	}
	ReadLiteral extends Read{
		//the literal is already in the Mem
		FIXME the literal should be written first so it gets write locked,
		so there should be WriteLiteral and WriteDynamic instead of 2 kinds of Read.
		Write will either use nonsandboxedLangColonCode or the literal
		as the thing to write.
		Should Write have a second Mem to
		copy from (abstractly, actually its a noOp)?
	}
	ReadDynamic extends Read{
		Write child; //read lock after write is finished
		constraint{ this.mem == this.child.mem }
	}
	
	//ForestOp{
	//	boolean isWrite; //Each op is either write or read.
	//	String nonsandboxedLangColonCode;
	//	Mem mem;
	//	ForestOp prevUseOfSameMemOrNull;
	//	List<ForestOp> waitChilds; //dependencies that arent used by nonsandboxedLangColonCode
	//	//List<ForestOp> computeChilds; //childs used by nonsandboxedLangColonCode
	//}
	The new logic is in waitChilds and prevUseOfSameMemOrNull.
	In examples I'll write each [prevUseOfSameMemOrNull kind of linkedlist].
	as a letter with number sequence like b0 b1 b2 and so on,
	and if theres no number it means 0,
	and I'll write y.waitChilds.contains(x) as x<-y,
	and I'll write y.computeChilds.contains(x) as x<=y.
	Its implied by the "letter with number sequence", but I can also write
	parts of the prevUseOfSameMemOrNull linkedlist like b0<-b1.
	...
	Example that uses the same CLMem in b0 and b2 which depend on eachother:
	a<=b0<=c<=b2
	b0<-lockB1<-b2
	c<-lockB1
	...
	Can that example be done without lockB1?
	a<=b0<=c<=b2 is a valid thing to do without any extra locking.
	The locks are designed to prevent nondeterminism of output
	while allowing nondeterminism of choosing any depending compatible order,
	where dependnet also describes when the same CLMem can be read and written.
	What if d<=b2 and not(b0<=d)? d and c now have some relation even though
	neither can reach the other through downward ptrs.
	If b0<=c then b0 is not finished until c finishes reading b0.
	...
	Redesign this so each CLMem alternates read/write,
	and evens (such as bR0 and bR2) are read and odds (such as bW1) are write.
	bW1<=bR2 means the copy op, but in practice its noOp cuz its the same CLMem,
	and this uses Id instead of CLMem so ForestOp is more general than opencl.
	aR0<=bW1<=bR2<-bW3
	bR2<=cW1<-cR2<=bW3
	
	
	
	
	
	
	
	copied to itsTimeToOpenclOptimizeRecurrentjavaWhichAtParallel5MayBeSlowerButAtParallel300WillBeFaster[
		ForestOp CLMem sharing solved this way[
		    There will be andother symbol in the forest, one for each clmem,
		    that means to free that clmem (from forestop, not at the opencl level
		    where it still exists), and that free can only be done when
		    all the direct parents of the forestop (that its a freeing of)
		    are done. That freeing op is therefore parent of all direct parents
		    of the thing its a freeing of.
		    For example, X and Y depend on W. freeW depends on X and Y.
		    Z depends on freeW which uses the same memory as W,
		    or more generally each is a sequence in how many times W's CLMem gets used.	    
		    
		]
		
		
		
		TODO multiple opencl ndrange kernels before returning to java,
		each in its own CLMem,
		and after thats working some system to run the forest
		that reuses CLMem objects where they are the same size
		in some cases such as adding into same array
		from multiple arrays that recurrentjava does
		using mutable arrays but here its modelled as immutable ops
		that can be optimized in mutable array locations.
	]
	
	//FIXME where does the n go in the output array is size n*howMany_getglobalid0?
	
	/*TODO Map<ForestOp<T>,String> and Map<String,CLMem>, OR Set<Set<ForestOp>>,
	for which ForestOps should share CLMem objects.
	*/
			
	/** leaf */
	public ForestOp(Class<T> ret){
		this(ret, "ForestOpLeaf:");
	}

	/** TODO derive Class<T> ret from nonsandboxedLangColonCode. If NonsandboxedOpenclNdrangeKernel, its the first param's type. */
	public ForestOp(Class<T> ret, String nonsandboxedLangColonCode, ForestOp... childs){
		super(nonsandboxedLangColonCode, null); //FIXME parallelSize of null. Added parallelSize for DependOp which is my sibling. ForestOp will likely become obsolete after OpenclUtil.callOpenclDependnet works.
		this.ret = ret;
		this.childs = Collections.unmodifiableList(new ArrayList(Arrays.asList(childs))); //immutable copy pointers
	}
	
	public final Class<T> ret;
	
	/** If nonsandboxedLangColonCode.startsWith("NonsandboxedOpenclNdrangeKernel")
	then childs.size() is ndRange.length (normally 1, later will also support 2 and 3) bigger than
	the number of opencl ndrange kernel params. Its the get_global_id(dimIndex) Integers.
	At least for now, its only 1d aka code contains "get_global_id(0)" and ndRange.length==1.
	<br><br>
	If nonsandboxedLangColonCode.startsWith("ForestOpLeaf"), this is empty and nonsandboxedLangColonCode.equals("ForestOpLeaf:")
	aka empty code string since the value is given instead of computed from childs.
	<br><br>
	TODO If nonsandboxedLangColonCode.startsWith("NonsandboxedJavaFunc"), childs are the same number of params
	as the java func's params. NonsandboxedJavaFunc will probably not be implemented for a long time
	since opencl is the main usecase. 
	*/
	public final List<ForestOp<?>> childs;
	
	/* TODO implement this first the inefficient way using OpenclUtil's existing ability
	to call 1 opencl ndrange kernel at a time then copy back to java,
	THEN implement GRU neuralnet with it (similar to recurrentjava) and verify it works
	in the newest existing mouseai experiment which has 3 waves per row
	(input, correctOutput, observedOutput),
	THEN optimize by using 1 CLMem per ForestOp and doing it all at once before returning to java
	(generalized for any forest of opencl ForestOps).
	
	TODO some params depend on other params in the forest, such as the size of a resulting matrix multiply.
	is not the same size as any of the param arrays, and that size affects sizes later in the forest
	which take that matmuled result as param.
	
	TODO derive array sizes from other array sizes which are const params in opencl kernels
	but within a call of a forest of OpenclForestOp can be variable until start the call.
	
	TODO update OpenclUtil to take a set or list of OpenclForestOp to get the array from
	and the rest are temp calculations that dont get copied to java (are in CLMem only),
	and the inputs are copied from java.
	
	TODO testcases of OpenclForestOp.
	
	TODO what is the param of each OpenclForestOp? Is it a Map whose values
	are the inputs (arrays, floats, ints, etc)?
			
	The (float,float)->float autodiff backprop will be manually coded in groups that are 1 kernel each
	so metarnnNodeType_2019-7-31-7a.jpg isnt needed (is hardcoded that way instead of generalizing,
	so its diff but not autodiff). It will be lower lag by grouping multiple such ops together
	instead of doing a separate matrix op for each.
	Remember GRU is already learning the chuasCircuit sample dataseqs very accurately
	and I'm ready to opencl optimize it before moving on to mouseai.
	*
	
	public String toString(){
		return "[ForestOp(childs="+childs.size()+"):"+nonsandboxedLangColonCode+"]";
	}*/

}
