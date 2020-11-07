package immutable.recurrentjava.flop.unary;

public class SigmoidUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return (float)(1 / (1 + Math.exp(-x)));
	}

	//benrayfield renamed this to deriv
	@Override
	public float deriv(float x) {
		float act = forward(x);
		return act * (1 - act);
	}
	
	//benrayfield added this
	public static final SigmoidUnit instance = new SigmoidUnit();
}
