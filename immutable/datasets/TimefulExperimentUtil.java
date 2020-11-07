/** Ben F Rayfield offers this software opensource MIT license */
package immutable.datasets;

/** Timeless means without sequence, such as RBM etc naturally learns. Timeful is what RNN etc tends to learn. */
public class TimefulExperimentUtil{
	
	/*public static WHAT trainingData(String dataset, ...);
	
	what should be the type of lstm trainintData?
	Example lstm training data: .4*Math.sin(time*5)+.3*Math.sin(time*2) for time steps in some range,
	and vary those constants in other trainingdata in the same dataset.
	Im not including interactive datasets in this model since that should be done
	only after the const datasets are learned well.
	Recurrentjava defines them as a List of <input matrix, correct output matrix>.
	But I might want to compress where data points are repeated
	such as input the mouse position at now-3 seconds, now-2.97 ..a. now-.3,
	and output the mouse at now, so if we slide forward by .03 seconds each time,
	many of those would overlap. It could be stored as a sequence of mouse positions
	with the chaostime offset in units of timesteps.
	There could be multiple dimensions such as mouseYFraction and mouseXFraction,
	so float[2][timestep] with chaostimesteps=20 forExample.
	But we need a float[2][timestep] for each such sequence, so..a.
	float[whichExample][2][timestep] with int chaostime.
	From "float[whichExample][2][timestep] with int chaostime" with a time offset and whichDim for
	each input and output, you generate for each whichExample
	a sequence of <set of inputs and set of outputs>.
	Its simplest to display if the outputs are always the NOW of each dim,
	and thats how Im planning to do it in lstm so maybe I should use that standard here.
	Also I might want an attention for each since I dont expect lstm to converge to the data
	after just 1 timestep, but I do expect it after some time steps to reproduce
	remaining data closely.
	*/

}