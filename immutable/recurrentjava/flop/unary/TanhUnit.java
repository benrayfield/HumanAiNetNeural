package immutable.recurrentjava.flop.unary;

public class TanhUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public double forward(double x) {
		return Math.tanh(x);
	}

	@Override
	public double deriv(double x) {
		//benrayfield's TODO??? or would it lose precision???...
		//optimize by tanh(x)=sigmoid(2*x)*2-1 aka tanh(x) = (1/(1+e^-(2*x)) * 2 - 1)
		//Sigmoid's derivative just calls exp once and the rest are fast ops,
		//compared to this calling cosh twice.
		double coshx = Math.cosh(x);
		double denom = (Math.cosh(2*x) + 1);
		return 4 * coshx * coshx / (denom * denom);
	}
}
