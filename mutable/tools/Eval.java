package mutable.tools;
import static mutable.util.Lg.*;
import java.awt.Component;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.util.function.Function;

import javax.swing.JLabel;

import bsh.EvalError;
import bsh.Interpreter;
import bsh.TargetError;
import immutable.util.Text;
import mutable.listweb.ListwebRoot;
import mutable.listweb.start.StartSingleUserWindow;
import mutable.uitool.Uitool;
import mutable.uitool.util.DisplayString;
import mutable.util.Var;

/** In a window with listweb on left and a TaskPlayerUi (UPDATE: a ResearchPanel) on right,
chooses what to put in the TaskPlayerUi at each moment based on whats in the
listweb defs (the low textarea), where every listweb node has a prilist and a def
and sometimes other properties, in JsonDS datastruct and permanently stored.
For example, try to get RbmEditor to learn a certain dataset or interactive
mouse experiment, by writing some stuff in that textarea,
including a Learnloop script.
*/
public class Eval{
	private Eval(){}
	
	/** TODO look in jars in data/lib, and make sure to set native lib path to data/lib
	before do this first time during JVM run (for lwjgl opencl etc).
	*/
	public static Class findClass(String name){
		try{
			return Class.forName(name);
		}catch(ClassNotFoundException e){
			throw new Error(e);
		}
	}
	
	/** potentially stateful.
	TODO put a textarea at bottom of the research area (TaskPlayerUi)
	which controls what happens in the research,
	so theres a looser connection between that and listweb.
	What do I want listweb to do?
	I want to store learnloop scripts in listweb so I and others can
	watch or do experiments that I consider to be useful progress toward mmgMouseai.
	But I also want the scripting language to do more than just RbmEditor.
	Maybe the first token should be a javaclass and the rest of the string
	is param to that javaclass? In that model, the rest of the string could refer to
	mindmap names which might contain other parts of the experiments,
	such as the mindmapname of a RBM state stored on harddrive
	or of a dataset. name it uitool:a.b.c so know the diff between that
	and other protocols such as occamsfuncer:$345345356345..345345.
	*/
	public static Object $(Object param){
		if(param instanceof String){
			String code = (String)param;
			code = code.trim();
			//if(code.startsWith("java:")){
			//	String javaCode = code.substring("java:".length()).trim();
			//	return evalJava(javaCode);
			if(code.startsWith("uitool:")){
				String firstToken = code.split("\\s")[0];
				String javaclassName = firstToken.substring("uitool:".length());
				//String restOfString = code.substring(firstToken.length()).trim();
				Class c = findClass(javaclassName);
				Constructor cn = null;
				try{
					cn = c.getConstructor();
					//return c.getConstructor(String.class).newInstance(restOfString);
				}catch(NoSuchMethodException e){
					//th 	row new Error("No constructor in "+c+" that takes a String");
					throw new Error("No parameterless constructor in "+c);
				}
				try{
					Uitool u = (Uitool) cn.newInstance();
					u.accept(code);
					return u;
				}catch(InvocationTargetException | IllegalAccessException | InstantiationException e){
					throw new Error("Error running constructor "+cn+". ", e);
				}
			}else if(code.startsWith("java:")){
				String javaCode = code.substring("java:".length()).trim();
				//String javaCode = code;
				Object ret = evalJava(javaCode);
				lg("return: "+ret);
				return ret;
				//throw new Error("ERROR_IN_THIS_CODE: "+code);
			}else if(code.startsWith("javaThread:")){ //run java code in new thread
				String javaCode = code.substring("javaThread:".length()).trim();
				new Thread(()->{
					evalJava(javaCode);
				}).start();
				return "New thread started (this text normally wont update when it ends or for errors, see stdout/stderr for that) with javaCode:\r\n"+javaCode;
			}else{
				Uitool u = new DisplayString();
				String message = code.trim().length()==0 ? "" : "I dont know what to do with:\r\n"+code;
				u.accept(message);
				return u;
				//throw new Error("Unknown code: "+code);
			}
		}else{
			return param;
		}
	}
	
	/** TODO beanshell can call this to do a variety of things.
	Its the only func I should allow beanshell to call other than funcs defined inside beanshell?
	*
	public static Object $(Object param){
		if(param instanceof String){
			return eval((String)param);
		}
		return param;
		//return "param is: "+param+" this is a test.";
	}*/
	
	static Object evalJava(String javaCode){
		lg("START evalJava: "+javaCode);
		try{
			Interpreter i = new Interpreter();
			//i.set("$", Eval.class.getMethod("$", Object.class));
			i.eval("$(p){ "+Eval.class.getName()+".$(p); };"); //create $ func in beanshell
			Object o = i.eval(javaCode);
			lg("END evalJava: "+javaCode);
			return o;
		}catch(TargetError t){
			lg("ENDBYTHROW evalJava: "+javaCode);
			Throwable targ = t.getTarget();
			if(targ instanceof Error) throw (Error)targ;
			if(targ instanceof RuntimeException) throw (RuntimeException)targ;
			throw new RuntimeException(targ);
		}catch(Throwable t){
			lg("ENDBYTHROW evalJava: "+javaCode);
			throw new Error(t);
		}
	}
	
	public static Uitool evalToUitool(String code){
		return ResearchPanel.asUitool($(code));
	}
	
	public static String exampleToEval = "FIXME, jlabel isnt a uitool... uitool:"+JLabel.class.getName()+" This is exampleToEval.";
	
	/*public static void nextState(){
		
		String mindmapName = StartSingleUserWindow.select;
		String def = ListwebRoot.def(mindmapName);
		
	}*/
	

}
