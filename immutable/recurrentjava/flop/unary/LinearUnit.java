package immutable.recurrentjava.flop.unary;

public class LinearUnit implements Unaflop {
	

	private static final long serialVersionUID = 1L;

	@Override
	public double forward(double x) {
		return x;
	}

	@Override
	public double deriv(double x) {
		return 1.0;
	}
	
	//benrayfield added this
	public static final LinearUnit instance = new LinearUnit();
}
