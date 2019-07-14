package mutable.listweb.todoKeepOnlyWhatUsingIn.zing;

/** Translates between zing (language) and object */
public interface Lang<T>{
	
	public T o(Zing z);
	
	public Zing z(T ob);

}
