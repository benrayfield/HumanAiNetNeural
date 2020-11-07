package immutable.forestop;

/** doesnt include eltype (such as float.class or long.class).
Memory is always 1d, so no 2d+ arrays.
Can copy to 2d array outside the ForestOp system if you want.
*/
public enum MemType{
	
	/** like float[] */
	arrayMem,
	
	/** like FloatBuffer */
	nioMem,
	
	/** like CLMem */
	gpuMem

}