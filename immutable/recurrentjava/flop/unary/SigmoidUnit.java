package immutable.recurrentjava.flop.unary;

public class SigmoidUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public double forward(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	//benrayfield renamed this to deriv
	@Override
	public double deriv(double x) {
		double act = forward(x);
		return act * (1 - act);
	}
	
	//benrayfield added this
	public static final SigmoidUnit instance = new SigmoidUnit();
}
