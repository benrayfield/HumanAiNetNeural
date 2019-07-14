package immutable.recurrentjava.flop.unary;

public class SineUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public double forward(double x) {
		return Math.sin(x);
	}

	@Override
	public double deriv(double x) {
		return Math.cos(x);
	}
}
