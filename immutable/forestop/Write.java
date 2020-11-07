package immutable.forestop;

/** !ForestOp.isWrite() */
public interface Write extends MemReadOp{
	
	/** lazy create and reuse a Read thats after this Write */
	public Read read();
}