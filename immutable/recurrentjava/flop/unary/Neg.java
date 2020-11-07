package immutable.recurrentjava.flop.unary;

/** class by benrayfield */
public class Neg implements Unaflop {

	private static final long serialVersionUID = 1L;

	public float forward(float x){
		return -x;
	}

	public float deriv(float x){
		return -1;
	}
}
