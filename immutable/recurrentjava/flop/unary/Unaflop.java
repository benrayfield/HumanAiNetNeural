package immutable.recurrentjava.flop.unary;

import java.io.Serializable;

/** benrayfield renamed Nonlinearity to Unaflop (unary floating point op),
and new class Biflop (such as multiply and add) for opencl optimization
redesigning the Graph class to do less and moving some of those ops
into an opencl kernel which will be compiled from these objects.
*/
public interface Unaflop extends Serializable{
	//benrayfield made these public
	public double forward(double x);
	public double deriv(double x);
	
	//benrayfield added float versions instead of double, for opencl optimization
	public default float forward(float x){
		return (float)forward((double)x);
	}
	
	public default float deriv(float x){
		return (float)deriv((double)x);
	}
}
