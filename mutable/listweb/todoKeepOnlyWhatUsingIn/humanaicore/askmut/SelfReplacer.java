package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut;

public interface SelfReplacer<T extends SelfReplacer<T>>{
	
	/** Returns a backed view of self that this object prefers callers use instead,
	maybe because of optimizations done after it was created and it wants to garbcol the old,
	which is probably only useful for at least partially immutable objects.
	A backed view of an immutable object does not necessarily point at that object.
	*/
	public T trySelfReplace();

}
