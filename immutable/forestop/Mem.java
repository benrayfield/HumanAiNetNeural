package immutable.forestop;
import java.io.Closeable;
import java.io.IOException;

public interface Mem extends Closeable{
	
	/** Example: float.class byte.class int.class */
	public Class eltype();
	
	public int size();
	
	public void close();

}
