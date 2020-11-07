package mutable.util;

/** WARNING: things will silently fail without obvious reason if you dont know to call this after adding work.
lazyEvals things only when manually call this function, instead of automatically when observed
which would otherwise silently get the wrong data or throw.
For example, its slowe to subclass FloatBuffer to check if its content should be modified
in every get and set call, so OpenclGraph reads and writes many FloatBuffer in many Matrix,
if it has any work to do. You can check if there is work to do.
All funcs here are synchronous even if the work inside it is asynchronous.
<br><br>
UPDATE: In lazycl (another software likely to be merged into this one) every read of float double int long etc
by a get func (which may wrap FloatBuffer or int[] etc) uses catch(NullPointerException)
which I've heard has hardware optimizations, to check if it needs to run eval() first.
*/
public interface ManualLazy{
	
	/** If true, then you should check hasWork() and if so doWork() will have an effect, such as in OpenclGraph.
	If false, hasWork() will always be false cuz work is done as soon as its added, such as in CpuGraph.
	*/
	public boolean isLazy();
	
	public boolean hasWork();
	
	public void doWork();

}
