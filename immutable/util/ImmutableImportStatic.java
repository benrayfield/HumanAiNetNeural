/** Ben F Rayfield offers this software opensource MIT license */
package immutable.util;
import java.util.*;
import java.util.function.*;

/** functions to import static many places, for immutable objects (though sometimes immutability cant be known, its callers responsibility) */
public class ImmutableImportStatic{
	
	public static <T> SortedSet<T> setAnd(SortedSet<T> x, Predicate<T> y){
		SortedSet<T> ret = new TreeSet();
		for(T item : x) if(y.test(item)) ret.add(item);
		return Collections.unmodifiableSortedSet(ret);
	}

}