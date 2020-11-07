package mutable.util;

import java.lang.reflect.Method;

import bsh.Interpreter;

public class Throw{
	
	public static void s(String s){
		throw new Error(s);
	}
	
	/** cuz it wasnt displaying the stacktrace in stderr in Eval.$ of a javaThread: that throws as of 2019-7-27 */
	public static void useThisFuncToTestStackTrace(String s){
		s(s);
	}
	
	public static void useThisFuncToTestStackTraceByJavaReflect() throws Throwable{
		Method m = Throw.class.getMethod("useThisFuncToTestStackTrace",String.class);
		m.invoke(null, "useThisFuncToTestStackTraceByJavaReflect");
	}
	
	public static void throwInBeanshellCode() throws Throwable{
		Interpreter i = new Interpreter();
		i.eval("mutable.util.Throw.useThisFuncToTestStackTrace(\"throwInBeanshellCode\");");
	}
		
	public static void main(String... args) throws Throwable{
		
		throwInBeanshellCode();
		
		//useThisFuncToTestStackTraceByJavaReflect();
		/** Correct, unlike the Eval.$ not displaying mutable.util.Throw.s:
		Exception in thread "main" java.lang.reflect.InvocationTargetException
		at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
		at sun.reflect.NativeMethodAccessorImpl.invoke(Unknown Source)
		at sun.reflect.DelegatingMethodAccessorImpl.invoke(Unknown Source)
		at java.lang.reflect.Method.invoke(Unknown Source)
		at mutable.util.Throw.useThisFuncToTestStackTraceByJavaReflect(Throw.java:18)
		at mutable.util.Throw.main(Throw.java:22)
		Caused by: java.lang.Error: useThisFuncToTestStackTraceByJavaReflect
		at mutable.util.Throw.s(Throw.java:8)
		at mutable.util.Throw.useThisFuncToTestStackTrace(Throw.java:13)
		... 6 more
		*/
	}

}
