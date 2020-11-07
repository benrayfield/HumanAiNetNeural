package immutable.forestop;
import java.sql.Time;
import java.util.List;

public interface MemReadOp extends ForestOp_deprecated{
	
	/** This is either a read lock or write lock.
	Except for the first write,
	...read(x0)<-write(x1)<-read(x2)<-write(x3)...
	and all those are wait edges, between the same Mem x.
	*/
	public boolean isWrite();

	/** true if this is the first ForestOp which can touch a certain Mem,
	which may be either a literal written from outside the ForestOp system
	or code() called on computeChilds().
	There can be only 1 ForestOp which isFirst() for each Mem.
	Every first is a write.
	In each running of a forest of ForestOp,
	Every Mem must have 1 such ForestOp which isFirst().
	Those Mems can be used again,
	such as not freeing CLMems after opencl returns to java
	if the same size is expected to be used again
	and if it makes things faster.
	*/
	public boolean isFirst();
	
	/** If isWrite() then it can be literal already in Mem (from external)
	or code() called on computeChilds().
	*/
	public default boolean isLiteral(){
		return isWrite() && code()==null;
	}
	
	public Mem mem();
	
	/** If nonnull this is the code to write,
	given the List<ForestOp> of read childs.
	If null and isWrite() then Mem contains a literal
	that was written externally before the forest started.
	If null and !isWrite() then there is at most 1 child,
	and if there is 1 child its a read of the same mem(),
	and if there are 0 childs then this is the first write of mem().
	<br><br> 
	Example strings: "openclNdrangeKernel:"... or "java:"...
	*/
	public String code();
	
	/* public List<ForestOp> childs();
	FIXME If isWrite() then theres 1 child thats read of same mem()
	thats to prove all read locks of that mem() finished
	before writing it but does not actually read that mem().
	That is a wait edge.
	The 2 edge types are waitEdge and computeEdge.
	*/
	public List<ForestOp_deprecated> waitChilds();
	
	/** explained in comment of waitChilds() */
	public List<ForestOp_deprecated> computeChilds();
	
	/** Example: int or float. These params go after computeChilds in kernel code. *
	public List<Number> primChilds();
	//FIXME Foresting.public Write newWrite(Read waitToUnlockReadOfSameMem, String code, Read... computeChilds);
	*/
	
	/*TODO a forest builder that can only build allowed forests,
	other than it doesnt verify code() fits the sizes of computeChilds()
	but does verify code() only reads computeChilds() and writes its own Mem
	and that none of its childs have the same Mem,
	and it verifies theres a 1 to 1 mapping between isFirst() and Mem.
	Use currying to build that from immutable objects?
	Represent it in occamsfuncer using Op.nondet
	(later to implement in the 15 other formalVerified deterministic ops)?
	If so, should such funcs start with unsafeocfnplug
	so untrusted code cant call them like untrusted code
	is allowed to call funcs starting with ocfnplug?
	Put an option, that starts false, in ImportStatic or Gas,
	thats static boolean allowUnsafe.
	Have such a func for the various things checked about opencl ndrange
	kernel code such as what is its name, which params are read and which
	are write, etc. Occamsfuncer would have to be further optimized,
	and that could take a long Time. I want this working soon,
	so I should not use occamsfuncer yet,
	but I still need a forest builder that cant build a forest
	that violates read write locking or other constraints
	but the string code() is not verified to work with the Mem sizes.
	
	What ops would this builder have?
	Each builder op returns a pair<Set<ForestOp>,ForestOp>
	or maybe just a ForestOp.
	The Set<ForestOp> might be needed to verify 1 to 1 mapping 
	between Mem and isFirst.
	Instead of Set<ForestOp> it could be List<ForestOp>
	in any dependnet order.
	So every builder op returns a List<ForestOp>
	and the ForestOp added last is at end of list,
	and different builder ops may take other params.
	List<ForestOp> builderOp(List<ForestOp> prev, Object... otherParams)?
	No, its confusing what params go in otherParams
	and how to access specific Mem objects or create a new Mem object.
	...
	These are the few kinds of builder ops:
	-- create empty forest.
	-- create isFirst of literal, which is a Write.
	-- create Read with 1 Write child of same Mem.
	-- create Write that uses code() on List<ForestOp> computeChilds
		and has a size 1 List<ForestOp> waitChilds which is Read of same Mem,
		and mark this write as either to be returned externally or not.
	-- call forest on Map<ForestOp,Object>,
		only including keys for the literals (and Object is their value),
		and returning Map<ForestOp,Object> which copies Mem to Object,
		and this must not be done at the same time as reusing ForestOps or Mems
		so a simple (but not the best) way to do that is to make that
		a static synchronized java func.
		This is not a "builder" op but is a similar op so I will include it
		and name it something other than "builder".
	===Should Mem be abstracted to MemId and use Map<MemId,Mem> in "call forest" op?
		Or would you just make another forest to call it again?
	===Should copying between opencl and java (like done in OpenclUtil)
		use a copy op between 2 Mem?
		How should CLMem vs float[] vs FloatBuffer etc
		get attached to a specific Mem object or instantiate a Mem of those?
		I want CLMems to be allocated, reused, and freed automatically
		without the caller of builder ops having to know about it,
		so maybe builder ops should include the finding andor allocating
		of chosen memory type (opencl, *Buffer, prim java array, etc),
		eltype (float.class byte.class int.class etc),
		and size?
	*/

}

