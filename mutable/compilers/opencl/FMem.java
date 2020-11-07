package mutable.compilers.opencl;

import java.nio.FloatBuffer;

import org.lwjgl.opencl.CLMem;

public class FMem extends BiMem<FloatBuffer> implements Cloneable{
	
	public FMem(int size){
		super(float.class,size);
	}
	
	public FMem(FloatBuffer buf, CLMem c){
		super(buf,c);
	}
	
	/** read float at int index from the FloatBuffer,
	but this doesnt work if havent queued and executed an opencl action to sync
	from CLMem to FloatBuffer (see OpenclUtil for example,
	but TODO will have a function in Matrix andOr Graph for it.
	*/
	public final float get(int index){
		return buf.get(index);
	}
	
	/** write float at int index. See comment of get(int). */
	public final void put(int index, float f){
		buf.put(index,f);
	}
	
	/** write plusEqual f at index. See comment of get(int). */
	public final void putPlus(int index, float addMe){
		buf.put(index,buf.get(index)+addMe);
	}
	
	/** write multiplyEqual f at index. See comment of get(int). */
	public final void putMult(int index, float multMe){
		buf.put(index,buf.get(index)*multMe);
	}
	
	/** write divideEqual f at index. See comment of get(int).
	This is probably slightly more accurate than putMult(int, 1/multMe).
	*/
	public final void putDivide(int index, float divideMe){
		buf.put(index,buf.get(index)/divideMe);
	}
	
	/** nonbacking. This wont work if the data is in the CLMem now
	and hasnt been queued and executed to sync to the FloatBuffer.
	*/
	public float[] toFloatArray(){
		float[] f = new float[buf.capacity()];
		for(int i=0; i<f.length; i++) f[i] = buf.get(i);
		return f;
	}
	
	public Object clone(){
		FMem f = new FMem(size);
		//TODO optimize
		for(int i=0; i<size; i++) f.buf.put(buf.get(i));
		return f;
	}
	
	/** Same params as System.arraycopy except for FloatBuffers.
	<br><br>
	TODO optimize by using FloatBuffer.put(FloatBuffer),
	and make sure to put their positions capacities etc
	back the way they were before the copy except the
	range thats been copied.
	*/
	public static void arraycopy(
			FloatBuffer from, int fromIndex, FloatBuffer to, int toIndex, int len){
		for(int i=0; i<len; i++){
			to.put(toIndex+i, from.get(fromIndex+i));
		}
	}

}
