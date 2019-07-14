package mutable.listweb.todoKeepOnlyWhatUsingIn.start;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.function.*;

import javax.swing.JFrame;

import mutable.listweb.ListwebRoot;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Cx;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Zing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ZingRoot;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl.*;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ui.ZingWindow;

public class Start{
	private Start(){}
	
	static{lg("TODO consider removing zing since theres still unifying to do with ufnode much later");}
	
	static{lg("TODO make all uses of NavigableMap/List/etc immutable, using a set(Object...) and get(Object...) funcs of tree path, before add rbmParams key to some mindmapItems and rbmcirclesui");}
	
	private static final BiFunction<Zing,Cx,Zing> rootStart = (z, c)->{
		//In this software's java lambdas,
		//params are always named z and c if Zing and Cx,
		//else x and y if they're both stateless. Cx is stateful as little as possible.
		
		Zing mainArgs = z.mapGet("mainArgs").hsah();
		//Zing lg = z.mapGet("fn").mapGet("lg");
		//c.funcall(lg,"Testing the logger (lg). mainArgs="+ZingRoot.toListOfString(mainArgs));
		//FIXME only call zing funcs from lambdas used with zing
		lg("mainArgs="+(mainArgs==null?null:ZingRoot.toListOfString(mainArgs)));
		//lg("getByHash mainArgs = "+ZingRoot.toListOfString(ZingRoot.get(mainArgs)));
		//throw new Todo();
		System.err.println("TODO BiFunction<Zing,Cx,Zing> rootStart (THIS IS OLD ZING STUFF, AND IM GOIGN WITH NAVIGABLEMAP/LIST/ETC FOR NOW)");
		return ZingRoot.emptyMap; //FIXME
	};
	
	static final boolean zingResearch = false;
	
	public static void main(String[] args){

		//if(!zingResearch){ //TODO this should be included in zingResearch after get more things working, but for now I turned off the ui
			//Lambdas shouldnt have access to ui except through zing, so creating it here
			ListwebRoot.boot();
			//try{
				JFrame window = new ZingWindow("Listweb "+ListwebRoot.rootDir, "defaultUpStack", "defaultDownStack", "defaultSelectPtr", "searchResults", true);
				//"TODO use Z.vm (the root zing virtual machine) and fork into different vms per thread and recursively to limit computeMoney and memoryMoney"
			//}finally{
			//	ListwebRoot.jsonLockFile.delete();
			//}
		//}
			
		if(zingResearch){
			Zing zingAbcdefghi = ZingRoot.s("abcdefghi");
			String stringAbcdefghi = ZingRoot.s(zingAbcdefghi);
			lg("Testing string/zing transform: "+stringAbcdefghi);
			Zing zingAbcdefghiabcdefghi = ZingRoot.s(stringAbcdefghi+stringAbcdefghi);
			lg("Testing string/zing transform2: "+ZingRoot.s(zingAbcdefghiabcdefghi));
			lg("Testing string/zing transform3: "+ZingRoot.s(ZingRoot.s("3")));
			Zing zingAbcdefghiabcdefghi2 = ZingRoot.s("abcdefghiabcdefghi");
			Zing zingAbcdefghiabcdefghj = ZingRoot.s("abcdefghiabcdefghj");
			lg("should equal: "+zingAbcdefghiabcdefghi.equals(zingAbcdefghiabcdefghi2));
			lg("should not equal: "+zingAbcdefghiabcdefghi.equals(zingAbcdefghiabcdefghj));
			Cx c = new SimpleCx();
			List<String> argsList = Arrays.asList(args);
			lg("mainArgsCorrect="+argsList);
			Zing mainArgs = ZingRoot.toZingFromListOfString(argsList);
			lg("mainArgs="+ZingRoot.toListOfString(mainArgs));
			Map<Zing,Zing> m = new HashMap();
			m.put(ZingRoot.s("mainArgs"), mainArgs);
			//m.put(ZingRoot.s("fn"), ZingRoot.defaultFuncs());
			Zing param = new LongArrayZing(m);
			rootStart.apply(param, c);
		}
		
	}
	
	/*
	//"The Collection stuff here is old. Replace it with Zing."
	
	private static final Map map, innerMap;
	static{
		Map m = new HashMap();
		innerMap = Collections.synchronizedMap(new HashMap());
		m.put("lg", (Consumer<String>)(s->System.out.println(s)));
		m.put("get", (Function)(key->innerMap.get(key)));
		m.put("put", (BiFunction)((key,value)->innerMap.put(key,value)));
		m.put("$m", "TODO put memoryMoney object here");
		m.put("$c", "TODO put computeMoney object here");
		m.put("fn", "TODO put function (or BiFunction?) here to get a Function by hashname");
		m.put("fni", "TODO put function (or BiFunction?) here to get an IntUnaryOperator by hashname");
		map = Collections.unmodifiableMap(m);
	}
	
	public static final Function rootContext = o->{
		return map.get(o);
	};
	
	public static final BiFunction<NavigableMap,Function,Object> rootStart = (tree,context)->{
		List mainArgs = (List) ((Function)context.apply("get")).apply("mainArgs");
		Consumer<String> lg = (Consumer)context.apply("lg");
		lg.accept("OLD: zingjson is pureImmutable json NavigableMaps/Lists/Strings/Zings/Doubles with mutable context Functions");
		lg.accept("OLD: TODO try to hardcode as little as possible and instead put code and funcs each in their own file by hashname where they can call and use eachother");
		throw new RuntimeException("TODO put main program loop here, including ui and neuralnets and mousemoveai. mainArgs="+mainArgs);
	};
	
	public static void main(String[] args){
		//Lambdas shouldnt have access to ui except through zing, so creating it here
		
		ListwebRoot.boot();
		try{
			new ZingWindow("HumanAiNet", "defaultUpStack", "defaultDownStack", "defaultSelectPtr", true);
			//"TODO use Z.vm (the root zing virtual machine) and fork into different vms per thread and recursively to limit computeMoney and memoryMoney"
		}finally{
			ListwebRoot.jsonLockFile.delete();
		}
		
		
		//TODO redesign the following code to use only Zing
		
		//TODO? ((BiFunction)rootContext.apply("put")).apply("mainArgs",Collections.unmodifiableList(Arrays.asList(args)));
		//TODO? rootStart.apply(Collections.emptyNavigableMap(), rootContext);
	}*/
}
