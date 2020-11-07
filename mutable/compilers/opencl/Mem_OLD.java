package mutable.compilers.opencl;
import java.io.Closeable;
import java.io.IOException;
import java.nio.Buffer;

import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLMem;

/** Wraps a LWJGL CLMem, double[], float[], int[], Float or Integer, etc.
If its a Float or Integer etc, its like one of the vararg params of OpenclUtil.callOpencl.
WARNING: CLMem and in some cases FloatBuffer etc are outside JVM memory,
and I'm undecided if my finalize func should release it or if caller should.
*/
public class Mem_OLD<T> implements Closeable{
	
	public Mem_OLD(T ptr){
		this.ptr = ptr;
	}
	
	public T ptr;
	
	public boolean ifCLMemThenFreeItOnClose = true;
	
	protected void finalize() throws Throwable{
		close();
	}

	public synchronized void close() throws IOException{
		if(ptr != null){
			if(ptr instanceof CLMem){
				CL10.clReleaseMemObject((CLMem)ptr);
			}else if(ptr instanceof Buffer){
				throw new Error("TODO");
			}
			ptr = null;
		}
	}

}
