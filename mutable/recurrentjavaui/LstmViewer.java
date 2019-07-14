package mutable.recurrentjavaui;
import java.awt.Graphics;
import javax.swing.JPanel;
import mutable.recurrentjava.model.NeuralNetwork;

public class LstmViewer extends JPanel{ //rename to LstmMouseAI
	
	public NeuralNetwork lstm;
	
	public LstmViewer(NeuralNetwork lstm){
		this.lstm = lstm;
	}
	
	public void paint(Graphics g){
		/*TODO put inputs and outputs as the same node visually,
		with hidden nodes, all on diagonal, and a square of horizontal is FROM and vertical is TO,
		and theres 4 colors (of varying brightness) of diagonal cuz 4 edgetypes,
		and 1 of vertical or horizontal will have that color from those nodes
		to show which kind of node it is TO. All FROMs are output nodes.
		Put mouse timewindow on the input and output nodes,
		and for invarep only input part of timewindow thats chaostime backward
		and have it predict the now and display those predictions in an overlapping spiralmouserecorder.
		Do this live without recording at first and add recording later. For now just stream
		it directly into the lstm. Upgrade to opencl relearning many prev times (of same duration) at once,
		and I want to see its predictions of all those old times after each learning batch
		which randomly selects some.
		*/
	}

}
