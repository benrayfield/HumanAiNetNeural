package immutable.recurrentjava.flop.binary;

public class Add implements Biflop{

	public double forward(double a, double b){
		return a+b;
	}

	public double derivA(double a, double b){
		return 1;
	}

	public double derivB(double a, double b){
		return 1;
	}
	
	public float forward(float a, float b){
		return a+b;
	}
	
	public float derivA(float a, float b){
		return 1;
	}
	
	public float derivB(float a, float b){
		return 1;
	}

}
