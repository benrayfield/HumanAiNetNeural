package mutable.compilers.opencl;
import java.nio.Buffer;
import java.nio.FloatBuffer;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLMem;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.DependParam;
import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.Mem;
import mutable.compilers.opencl.connectors.lwjgl.Lwjgl;

/** UPDATE: this is a replacement for float[] in recurrentjava Matrix
and always has a Buffer (such as FloatBuffer)
and lazyCreates a CLMem to use with it.
OLD...
This is a way to copy between java memory and opencl memory. 
Can allocate CLMem by copy_host_pointer to copy a FloatBuffer
and read back into the FloatBuffer by scheduling a read in CLQueue.
When this object is garbcoled, frees the CLMem.
TODO There will be a makeReadable(BiMem) func somewhere
in redesign of recurrentjava andOr forestop
used to put that read op in CLQueue. 
*/
public class BiMem<T extends Buffer> extends Mem{
	
	public T buf;
	
	private CLMem cl;
	public CLMem cl(){
		if(cl == null){
			if(elType == float.class){
				cl = Lwjgl.instance().clmemWrapsJavaMem((FloatBuffer)buf);
			}else{
				throw new Error("TODO? primType="+elType);
			}
		}
		return cl;
	}
	
	/** starts false. will be true after calling cl() */
	public boolean hasCLMem(){
		return cl != null;
	}
	
	/** size is in units of primitives, unlike in CLMem */
	public BiMem(Class elType, int length){
		super(elType, length);
		if(elType == float.class){
			buf = (T) BufferUtils.createFloatBuffer(length);
		}else{
			throw new Error("TODO? primType="+elType);
		}
	}
	
	/** fixme BiMem varies between lazy and nonlazy
	as its Buffer (such as FloatBuffer) may have the current values
	or values may be in CLMem waiting to sync to or from the Buffer,
	or the CLMem may not be allocated yet.
	After 2020-2-6+ a redesign is starting (TODO) where the CLMem in BiMem
	wont be used and insted only use CLMem thru OpenclUtil.callOpenclDepend
	which uses DependParam (which is lazy() and does not contain any memory
	to grab a CLMem from a pool while doing the whole recurrentjava Graph's
	3 lists of Task (forwardprop, backpropReversed, and trainprop)
	all in a single call from java to opencl to do multiple kernels then
	back to java for lower lag. In that design, FIXME, I'm still undecided
	where the outputs of penclUtil.callOpenclDepend will be stored,
	such as maybe replacing the MemInfos in each Matrix with another
	MemInfo which is not lazy, or maybe keeping those values in
	a WeakHashMap<MemInfo,whatType> outside of Matrix and outside of MemInfo?
	*/
	public boolean lazy(){
		if(!hasCLMem()) return false; //data is only in Buffer, no need to sync (yet)
		throw new Error("TODO unknown if this is lazy or not");
	}
	
	private static Class throwDontReturnClass(String s){
		throw new Error(s);
	}
	
	private static int throwDontReturnInt(String s){
		throw new Error(s);
	}
	
	public BiMem(T buf, CLMem cl){
		super(
			(buf instanceof FloatBuffer)
				? float.class
				: throwDontReturnClass("TODO "+buf.getClass()),
			(buf instanceof FloatBuffer)
				? buf.capacity()
				: throwDontReturnInt("TODO "+buf.getClass())
		);
		if(buf instanceof FloatBuffer){
			//elType = float.class;
			//length = buf.capacity();
		}else{
			throw new Error("TODO? bufType="+buf.getClass());
		}
		this.buf = buf;
		this.cl = cl;
	}
	
	protected void finalize() throws Throwable {
		buf = null;
		if(cl != null){
			CL10.clReleaseMemObject(cl);
			cl = null;
		}
	}

}
