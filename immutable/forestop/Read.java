package immutable.forestop;

/** ForestOp.isWrite() */
public interface Read extends ForestOp_deprecated{
	
	/** lazy create and reuse a Write thats after this Read */
	public Write write();
}
