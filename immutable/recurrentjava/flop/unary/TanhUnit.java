package immutable.recurrentjava.flop.unary;

public class TanhUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return (float)Math.tanh(x);
	}

	@Override
	public float deriv(float x) {
		//benrayfield's TODO??? or would it lose precision???...
		//optimize by tanh(x)=sigmoid(2*x)*2-1 aka tanh(x) = (1/(1+e^-(2*x)) * 2 - 1)
		//Sigmoid's derivative just calls exp once and the rest are fast ops,
		//compared to this calling cosh twice.
		double coshx = Math.cosh(x);
		double denom = (Math.cosh(2*x) + 1);
		return (float)(4 * coshx * coshx / (denom * denom));
	}
}
