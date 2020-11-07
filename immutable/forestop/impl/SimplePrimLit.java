package immutable.forestop.impl;

import immutable.forestop.PrimLit;
import immutable.forestop.Write;

public class SimplePrimLit implements PrimLit{
	
	public final Number lit;
	
	public SimplePrimLit(Number lit){
		this.lit = lit;
	}

	public Number lit(){
		return lit;
	}

	public Write write(){
		throw new UnsupportedOperationException("primitives are copy by value. cant overwrite a constant.");
	}

}
