package immutable.rnn.metarnn;

public enum Unaflop{
	
	neg,
	
	oneDiv,
	
	exp,
	
	log,
	
	sigmoid,
	
	/** an aftrans of sigmoid, where tanh(x)=x as x limits to 0 */
	tanh,
	
	sin,
	
	asin,
	
	cos,
	
	acos,
	
	sqrt,
	
	sq,
	
	cbrt,
	
	cb,
	
	expm1;
	
	

}
