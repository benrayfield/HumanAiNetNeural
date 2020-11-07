package immutable.forestop;

/** literal primitive, like int or float param in opencl ndrange kernel
thats copied by value instead of by CLMem pointer.
*/
public interface PrimLit extends Read{
	
	/** get the literal primitive */
	public Number lit();

}
