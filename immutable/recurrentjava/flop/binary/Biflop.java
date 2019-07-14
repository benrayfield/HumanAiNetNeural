package immutable.recurrentjava.flop.binary;

import java.io.Serializable;

/** added by benrayfield for moving some of the Graph ops into a block of ops done all at once,
especially as compiled to opencl and maybe also will have compiler to javassist.
*/
public interface Biflop extends Serializable{
	
	//TODO
	
	public double forward(double a, double b);
	
	public double derivA(double a, double b);
	
	public double derivB(double a, double b);
	
	public default float forward(float a, float b){
		return (float)forward((double)a, (double)b);
	}
	
	public default float derivA(float a, float b){
		return (float)derivA((double)a, (double)b);
	}
	
	public default float derivB(float a, float b){
		return (float)derivB((double)a, (double)b);
	}

}
