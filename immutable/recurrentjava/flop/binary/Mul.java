package immutable.recurrentjava.flop.binary;

public class Mul implements Biflop{

	public double forward(double a, double b){
		return a*b;
	}

	public double derivA(double a, double b){
		return b;
	}

	public double derivB(double a, double b){
		return a;
	}
	
	public float forward(float a, float b){
		return a*b;
	}
	
	public float derivA(float a, float b){
		return b;
	}
	
	public float derivB(float a, float b){
		return a;
	}

}
