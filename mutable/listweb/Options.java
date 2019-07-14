package mutable.listweb;
import java.util.Collections;
import java.util.NavigableMap;
import java.util.TreeMap;

import immutable.occamsjsonds.JsonDS;

public class Options{
	private Options(){}
	
	/*static NavigableMap options(){
		NavigableMap map = (NavigableMap) ListwebRoot.get(ListwebRoot.rootName).get("options");
		if(map == null){
			//map = new TreeMap(OccamsJsonDS.mapKeyComparator);
			map = Collections.emptyNavigableMap(); //immutable
			ListwebRoot.get(ListwebRoot.rootName).put("options", map);
			onOptionChanged();
		}
		return map;
	}*/
	
	/*static void onOptionChanged(){
		ListwebRoot.onChange(ListwebRoot.rootName);
	}*/
	
	public static void setOption(String name, boolean value){
		setOption(name, value?1:0);
	}
	
	public static void setOption(String name, double value){
		//options().put(name,value);
		ListwebRoot.setWithEvent(ListwebRoot.rootName, "options", name, value);
		//onOptionChanged();
	}
	
	public static void setOption(String name, String value){
		//options().put(name,value);
		//onOptionChanged();
		ListwebRoot.setWithEvent(ListwebRoot.rootName, "options", name, value);
	}
	
	public static boolean option(String name, boolean setValueIfNotExist){
		return 0<option(name,setValueIfNotExist?1:0);
	}
	
	//TODO merge duplicate code? Or would that be bigger code to have the exact same behaviors including ClassCastException?
	
	public static double option(String name, double setValueIfNotExist){
		Object val = ListwebRoot.getElseNull(ListwebRoot.rootName, "options", name);
		if(val != null) return (double)val;
		setOption(name, setValueIfNotExist);
		return setValueIfNotExist;
		/*Double d = (Double) options().getElseNull(name);
		if(d == null){
			setOption(name, setValueIfNotExist);
			return setValueIfNotExist;
		}
		return d;
		*/
	}
	
	public static String option(String name, String setValueIfNotExist){
		Object val = ListwebRoot.getElseNull(ListwebRoot.rootName, "options", name);
		if(val != null) return (String)val;
		setOption(name, setValueIfNotExist);
		return setValueIfNotExist;
		/*String s = (String) options().getElseNull(name);
		if(s == null){
			setOption(name, setValueIfNotExist);
			return setValueIfNotExist;
		}
		return s;
		*/
	}
	
}