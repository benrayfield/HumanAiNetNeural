package immutable.recurrentjava.flop.unary;

public class RectifiedLinearUnit implements Unaflop {

	private static final long serialVersionUID = 1L;
	private float slope;
	
	public RectifiedLinearUnit() {
		this.slope = 0;
	}
	
	public RectifiedLinearUnit(float slope) {
		this.slope = slope;
	}
	
	@Override
	public float forward(float x) {
		if (x >= 0) {
			return x;
		}
		else {
			return x * slope;
		}
	}

	@Override
	public float deriv(float x) {
		if (x >= 0) {
			return 1f;
		}
		else {
			return slope;
		}
	}
}
