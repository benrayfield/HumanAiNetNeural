package mutable.chuascircuit;

/** I plan to use a sim of chuasCircuit as LSTM testcase
cuz its very simple and chaotic.
 
http://www.chuacircuits.com/sim.php QUOTE
Chua circuit equations:
	   dx/dt = c1*(y-x-f(x)) 
	   dy/dt = c2*(x-y+z)
	   dz/dt = -c3*y
	   f(x) = m1*x+(m0-m1)/2*(|x+1|-|x-1|)

	Initial: X: 
	0.7
	 Y: 
	0
	 Z: 
	0
	 
	Default X: 0.7    Y: 0      Z: 0
	C1:  
	15.6
	 Default: 15.6
	C2:  
	1
	 Default: 1
	C3:  
	28
	 Default: 28
	M0: 
	-1.143
	 Default: -1.143
	M1: 
	-0.714
	 Default: -0.714
UNQUOTE.
*/
public class ChuasCircuit{
	
	public double x, y, z, m0, m1, c1, c2, c3;
	
	public void nextState(double dt){
		double dxOverDt = c1*(y-x-(m1*x+(m0-m1)/2*(Math.abs(x+1)-Math.abs(x-1))));
		double dyOverDt = c2*(x-y+z);
		double dzOverDt = -c3*y;
		x += dxOverDt*dt;
		y += dyOverDt*dt;
		z += dzOverDt*dt;
	}

}
