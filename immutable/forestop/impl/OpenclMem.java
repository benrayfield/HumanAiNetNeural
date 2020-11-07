package immutable.forestop.impl;
import java.io.IOException;
import java.lang.ref.WeakReference;

import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLMem;

import immutable.forestop.Mem;

public class OpenclMem implements Mem{
	
	public CLMem clmem;
	//public final WeakReference<CLMem> clmem;
	
	/** TODO can this be read from CLMem instead of storing it? */
	public final Class eltype;
	
	/** TODO can this be read from CLMem instead of storing it? */
	public final int size;
	
	public OpenclMem(CLMem clmem, Class eltype, int size){
		this.clmem = clmem;
		this.eltype = eltype;
		this.size = size;
	}

	public Class eltype(){
		return eltype;
	}

	public int size(){
		return size;
	}
	
	protected void finalize(){
		close();
	}

	public synchronized void close(){
		if(clmem != null){
			CL10.clReleaseMemObject(clmem);
			clmem = null;
		}
	}

}