package immutable.recurrentjava.flop.unary;

import java.io.Serializable;

/** benrayfield renamed Nonlinearity to Unaflop (unary floating point op),
and new class Biflop (such as multiply and add) for opencl optimization
redesigning the Graph class to do less and moving some of those ops
into an opencl kernel which will be compiled from these objects.
*/
public interface Unaflop extends Serializable{
	//benrayfield made these public
	public float forward(float x);
	public float deriv(float x);
}
