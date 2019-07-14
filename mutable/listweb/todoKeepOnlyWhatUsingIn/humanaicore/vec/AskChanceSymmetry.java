package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec;

public interface AskChanceSymmetry extends BitVsScalar{
	
	/** Examples: True if neuralnet. False if bayes.
	If true, normally 0 has no effect and 1 has the most effect,
	so if any random subset of indexs are set to 0, an associativeMemory should still work.
	If false, the associativeMemory would pay equal attention to them being 0 vs 1.
	True uses them as attention (only true matters). False uses them as general chances.
	*/
	public boolean chancesAreAttention();
	
}