package immutable.recurrentjava.flop.unary;

public class LinearUnit implements Unaflop {
	

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return x;
	}

	@Override
	public float deriv(float x) {
		return 1f;
	}
	
	//benrayfield added this
	public static final LinearUnit instance = new LinearUnit();
}
