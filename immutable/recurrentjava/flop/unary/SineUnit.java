package immutable.recurrentjava.flop.unary;

public class SineUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return (float)Math.sin(x);
	}

	@Override
	public float deriv(float x) {
		return (float)Math.cos(x);
	}
}
