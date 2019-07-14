package immutable.recurrentjava.flop.unary;

/** class by benrayfield */
public class Neg implements Unaflop {

	private static final long serialVersionUID = 1L;

	public double forward(double x){
		return -x;
	}

	public double deriv(double x){
		return -1;
	}
}
