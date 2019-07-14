package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.test;
import java.util.function.Function;
//import javassist.ClassPool;
//import javassist.CtClass;
//import javassist.CtMethod;

/** Testing javassisst's ability to implement java lambda interface directly instead of the new syntax */
public class TestJavaLambda{
	public static void main(String[] args) throws Exception{
		/*new Function<Integer, Double>(){
            public Double apply(Integer i){
                return new Double(i.doubleValue()*10);
            }
        }*/
		
		/*String classPartCodes[] = {
			"public class SomeLambda implements java.util.function.Function",
			"public java.lang.Double apply(java.lang.Integer i){"
			+"\nreturn new java.lang.Double(i.doubleValue()*10);"
			+"\n}"
		};
		Class newClass = JavaHome.newClass(classPartCodes);
		*/
		
		
		/* TODO INCLUDE JAVASSIST
		ClassLoader cl = TestJavaLambda.class.getClassLoader();
		ClassPool p = ClassPool.getDefault();
		CtClass obCtClass = p.get("java.lang.Object");
		CtClass newCtClass = p.makeClass("SomeLambda", obCtClass);
		//CtClass newCtClass = p.makeClass("SomeLambda", p.getCtClass(gigalinecompile.test.Func.class.getName()));
		String javaFuncCode =
			"public java.lang.Object apply(java.lang.Object i){"
			//+"\nSystem.out.println((java.lang.Object y)->{ return y.toString(); }).apply(new java.lang.Double(33.));"
			+"\nreturn new java.lang.Double(((java.lang.Number)i).doubleValue()*10);"
			+"\n}";
		newCtClass.addMethod(CtMethod.make(javaFuncCode, newCtClass));
		
		//CtMethod ctApply = newCtClass.getMethod("apply", Descriptor.ofMethod(obCtClass, new CtClass[]{obCtClass}));
		//ctApply.setBody(CtMethod.make(javaFuncCode, newCtClass), null);
		newCtClass.addInterface(p.get("java.util.function.Function"));
		Class newClass = newCtClass.toClass(cl, null);
		
		
		Object o = newClass.newInstance();
		//Method apply = o.getClass().getMethod("apply", Object.class);
		//System.out.println("reflected: "+apply);
		//System.out.println("by reflect: "+apply.invoke(o, new Integer(7)));
		//System.out.println("is Function?: "+(o instanceof Function));
		Function f = (Function)o;
		System.out.println("by Function: "+f.apply(new Integer(55)));
		System.out.println("with an andThen: "+f.andThen((x)->((Number)x).doubleValue()+1000).apply(44));
		*/
	}
}