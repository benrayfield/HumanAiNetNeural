package immutable.acyclicflow;

/** a func of vector to vector, with a specific number of inputs and a specific number of outputs.
This can represent new kinds of neuralnodes (like in recurrentjava) compiled to opencl or javassist
or musical instruments (like in audivolv) or smooth cellular automata (like in smoothlife or physicsmata).
*/
public class AcyclicFlowF{
	
	public final int ins, outs;
	
	/** sequence */
	protected final int[] ops;
	
	public AcyclicFlowF(int ins, int[] ops, int outs){
		this.ins = ins;
		this.ops = ops.clone();
		this.outs = outs;
	}
	
	//TODO compile to javassist.
	
	//TODO compile to opencl.
	
	public int hashCode(){
		throw new Error("TODO");
	}
	
	public boolean equals(Object obj){
		throw new Error("TODO");
	}
	
	/*
	//FIXME should this be implemented as DependOp where the code might be "*:" or "+:" etc,
	//but that seems overcomplicated since that was designed for matrixs and this is designed for floats or doubles.
	
	TODO
	
	This class must have efficient hashCode and equals, maybe using securehash of the forest shape to test for equality,
	cuz Graph.java will use this to compile and lookup CompiledKernels.
	
	How about codesimian-like syntax such as sqrt(+(*(?#x x) *(?#y y)))#hypot or unlambda-like syntax but with ways to
	share branches by renaming, like ```*xx``*yy
	
	How about the acylicFlowInt12.12.8 existing code, with 2 ints for input size and output size.
	
	//TODO should this use Occamsfuncer? Occamsfuncer can represent acyclicflow, but theres alot of work to do there first.
	*/

}
