package immutable.forestop.impl;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

import immutable.forestop.Mem;

/** normally in cpu memory, but I dont know Buffer well enough to say
there arent any gpu forms of it.
*/
public class BufferMem<T extends Buffer> implements Mem{ //TODO rename to NioMem
	
	public T buf;
	
	public final Class eltype;
	
	public BufferMem(T buf){
		this.buf = buf;
		eltype = eltype(buf);
	}

	public Class eltype(){
		return eltype;
	}

	public int size(){
		return buf.capacity();
	}
	
	public static Class eltype(Buffer b){
		if(b instanceof FloatBuffer){
			return float.class;
		}else if(b instanceof DoubleBuffer){
			return double.class;
		}else if(b instanceof IntBuffer){
			return int.class;
		}else if(b instanceof LongBuffer){
			return long.class;
		}else if(b instanceof ShortBuffer){
			return short.class;
		}else if(b instanceof CharBuffer){
			return char.class;
		}else if(b instanceof ByteBuffer){
			return byte.class;
		}else{
			throw new Error("Unknown eltype of Buffer="+b);
		}
	}

	public void close(){
		buf = null;
	}

}
