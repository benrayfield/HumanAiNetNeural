package immutable.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ListUtil{ //TODO rename to ImListUtil meaning immutable list util
	
	/** returns immutable list */
	public static <T> List<T> cat(List<T>... lists){
		List<T> ret = new ArrayList();
		for(List<T> a : lists) ret.addAll(a);
		return Collections.unmodifiableList(ret);
	}
	
	public static <T> List<T> reverse(List<T> list){
		List<T> ret = new ArrayList(list);
		Collections.reverse(list);
		return Collections.unmodifiableList(ret);
	}
}
