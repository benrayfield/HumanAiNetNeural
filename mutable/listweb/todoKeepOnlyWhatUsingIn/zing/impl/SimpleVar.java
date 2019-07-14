package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Var;

public class SimpleVar<T> implements Var<T>{
	
	protected T val;
	
	public SimpleVar(T val){
		set(val);
	}
	
	public void set(T val){
		this.val = val;
	}
	
	public T get(){ return val; }

}
