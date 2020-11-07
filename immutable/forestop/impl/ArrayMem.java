package immutable.forestop.impl;
import java.io.IOException;
import java.lang.reflect.Array;

import immutable.forestop.Mem;

/** cpu, not gpu */
public class ArrayMem<T> implements Mem{
	
	public T array;
	
	public ArrayMem(T array){
		this.array = array;
	}

	public Class eltype(){
		return array.getClass().getComponentType();
	}

	public int size(){
		return Array.getLength(array);
	}

	public void close(){
		array = null;
	}

}
