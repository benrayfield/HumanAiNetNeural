package immutable.forestop;
import java.io.IOException;
import java.io.Closeable;

/** Builds and runs a dependnet that does write-one-read-many-locking,
such as a forest of ops that copy java arrays into opencl,
call multiple opencl ndrange kernels, then copy some arrays back to java.
The building and computing of a forest of ForestOps,
including the allocation of
opencl CLMem, java FloatBuffer, java float[], etc,
and ops for copying between them,
(TODO) optimized to do multiple opencl ndrange kernels
before returning to java (in theory will be lower lag),
and where hardware allows, using multiple opencl CLQueue
for dependnet order so can run a forest async
with the Foresting.run(...) calculation wrapping
an async forest in a sync call
(TODO which design pattern is that? Its similar to how
javascript is sync but can run async things in certain
sync calls such as receiving multiple ajax).
*/
public interface Foresting extends Closeable{
	
	/** make this an empty forest. Can be closed multiple times. */
	public void close();
	
	/** TODO: THIS WILL BE DONE AUTOMATICALLY BY THE OPS THAT RETURN Read or Write.
	<br><br>
	For example, may create a new CLMem or may reuse
	an existing CLMem that no other Foresting has locked,
	or may be java FloatBuffer or float[] or IntBuffer or int[] etc.
	Size is in units of eltype such as byte[3] or float[3].
	<br><br>
	All such mems are freed at the Foresting level when isDone()
	but not necessarily freed at the opencl level
	but will be automatically freed vs reused
	depending on optimizations. Caller does not need to free them.
	*/
	public Mem newMem(MemType memType, Class eltype, int size);
	
	public Mem newMem(MemType memType, Object literal);
	
	/** Example: wrap float[] or FloatBuffer in Mem *
	public Write newWriteLiteral(Object literal);
	*/
	
	/** create Read with 1 Write child of same Mem.
	There is at most 1 Read for each Write, and multiple Writes can
	simultaneously read that 1 Read, which forms a dependnet
	that enforces write-one-read-many-locking.
	*/
	public Read newRead(Write w);
	
	public Read newRead(Number literal);
	
	/** create Write that uses code() on List<ForestOp> computeChilds
	and has a size 1 List<ForestOp> waitChilds which is Read of same Mem,
	and mark this write as either to be returned externally or not.
	<br><br>
	code is language colon codeInThatLanguage such as "openclNdrangeKernel:..."
	and OpenclProgs has some example codeInThatLanguage.
	*/
	public Write newWrite(Read waitToUnlockReadOfSameMem, String code, Read... computeChilds);
	
	/** does the reading, computing, and writing in the Mems,
	then sets isDone() to true until the next clear().
	*/
	public void run();
	
	public boolean isDone();

}
