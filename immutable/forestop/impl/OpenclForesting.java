package immutable.forestop.impl;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.*;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLMem;

import immutable.forestop.Foresting;
import immutable.forestop.Mem;
import immutable.forestop.MemType;
import immutable.forestop.Read;
import immutable.forestop.Write;
import mutable.compilers.opencl.connectors.lwjgl.Lwjgl;

/** supports java and LWJGL (opencl/gpu) mem types,
such as copying float[]s into opencl,
computing multiple ndrange kernels,
then copying some of those to float[].
*/
public class OpenclForesting implements Foresting{
	
	/** ptrs to all Mems created so they dont get garbcoled
	if caller doesnt keep a ptr (TODO is that needed, since caller would
	still have ptr to ForestOp that they want to get the mems from) to them
	and (this is the main reason) so those that need freeing
	will be automatically freed on close() (which can occur multiple times)
	and close is also called during finalize().
	*/
	protected Set<Mem> mems = Collections.synchronizedSet(new HashSet());
	
	protected IntBuffer errorBuf;
	
	protected IntBuffer errorBuf(){
		if(errorBuf == null){
			errorBuf = BufferUtils.createIntBuffer(1);
		}
		return errorBuf;
	}
	
	public Mem newMem(MemType memType, Class eltype, int size){
		Mem m;
		switch(memType){
		case arrayMem:
			m = new ArrayMem(Array.newInstance(eltype, size));
		break;
		case nioMem:
			Buffer b;
			if(eltype == float.class){
				b = BufferUtils.createFloatBuffer(size);
			}else if(eltype == int.class){
				b = BufferUtils.createIntBuffer(size);
			}else{
				throw new Error("TODO memType="+memType+" eltype="+eltype);
			}
			m = new BufferMem<Buffer>(b);
		break;
		case gpuMem:
			CLMem clmem = CL10.clCreateBuffer(Lwjgl.instance().context(),
				CL10.CL_MEM_READ_ONLY, size*sizeInBytes(eltype), errorBuf());
			m = new OpenclMem(clmem, eltype, size);
			
			/*LWJGL copies FloatBuffer to CLMem at the same time as allocating CLMem,
			and my Mem interface needs to allocate those 2 things and copy later.
			Thats why there will be another newMem func that allocates and
			copies at once, for the inputs.
			*/
		default:
			throw new Error("Unknown memType: "+memType);
				
		}
		mems.add(m);
		return m;
	}
	
	/** Example: copy a FloatBuffer or float[] into a CLMem wrapped in an OpenclMem,
	since LWJGL requires copying inputs from java (but not between opencl and opencl)
	at time of allocating the CLMem (or at least queues it to do that?).
	*/
	public Mem newMem(MemType memType, Object literal){
		if(memType == MemType.gpuMem){
			Mem m = null;
			if(literal instanceof float[]){
				float[] f = (float[])literal;
				FloatBuffer b = BufferUtils.createFloatBuffer(f.length);
				CLMem clmem = CL10.clCreateBuffer(Lwjgl.instance().context(),
					CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_COPY_HOST_PTR, b, errorBuf);
				m = new OpenclMem(clmem, float.class, f.length);
			}else{
				throw new Error("TODO? literalType="+literal.getClass());
			}
			mems.add(m);
			return m;
		}else{
			throw new Error("TODO memType="+memType);
		}
	}
	
	static int sizeInBytes(Class primType){
		if(primType == double.class || primType == long.class){
			return 8;
		}else if(primType == float.class || primType == int.class){
			return 4;
		}else if(primType == short.class){
			return 8;
		}else if(primType == short.class || primType == char.class){
			return 2;
		}else if(primType == byte.class){
			return 1;
		}else{
			throw new Error("Not primitive: "+primType);
		}
	}
	
	/*public Write newWriteLiteral(Object literal){
		throw new Error("TODO");
	}*/
	
	public Read newRead(Write w){
		throw new Error("TODO");
	}
	
	public Write newWrite(Read waitToUnlockReadOfSameMem, String code, Read... computeChilds){
		throw new Error("TODO");
	}

	public void run(){
		throw new Error("TODO");
	}

	public boolean isDone(){
		throw new Error("TODO");
	}
	
	protected void finalize() throws Throwable{
		close();
	}

	public void close(){
		for(Mem m : mems){
			m.close();
		}
		mems.clear();
	}

	public Read newRead(Number literal){
		return new SimplePrimLit(literal);
	}

}