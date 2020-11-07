package mutable.mouseai;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

import immutable.util.MathUtil;
import mutable.mouseai.experiments.RecurrentjavaSingleChannelMouseai_todoParallel;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;

/** viewer for each of the float[3][timeCycles] views in mindmapname:unidimLstmEditor */
public class UnidimView extends JPanel{
	
	public static final int IN=0, CORRECT_OUT=1, OBSERVED_OUT=2, SIZE=3;
	
	public float[][] data;
	
	public UnidimView(int timeCycles){
		data = new float[SIZE][timeCycles];
		data[CORRECT_OUT][0] = .7f;
		data[CORRECT_OUT][1] = .8f;
		data[CORRECT_OUT][2] = .4f;
		setMinimumSize(new Dimension(50,50));
		setPreferredSize(new Dimension(50,50));
	}
	
	public static final Color inputColor = new Color(.7f, .2f, .2f);
	
	public static final  Color correctOutColor = new Color(.6f, .6f, 1f);
	
	public static final  Color observedOutColor = new Color(0f, .8f, 0f);
	
	/** displays scaled by bellcurve of all the data */
	public void paint(Graphics g){
		if(data.length != SIZE || data[0].length != data[1].length || data[0].length != data[2].length)
			throw new Error("Not a float[3][all same size]");
		g.setColor(Color.black);
		int w = getWidth(), h = getHeight();
		g.fillRect(0, 0, w, h);
		float[] cat = MathUtil.cat(data);
		float ave = MathUtil.ave(cat);
		float dev = MathUtil.devGivenAve(ave, cat);
		g.setColor(inputColor);
		//double displayDevs = 5;
		float displayDevs = 3;
		float mult = 1/(dev*displayDevs);
		paint(g, data[IN], w, h, ave, mult);
		g.setColor(correctOutColor);
		paint(g, data[CORRECT_OUT], w, h, ave, mult);
		g.setColor(observedOutColor);
		paint(g, data[OBSERVED_OUT], w, h, ave, mult);
	}
	
	public static void paint(Graphics g, float[] wave, int width, int height, float ave, float mult){
		for(int i=1; i<wave.length; i++){
			float fromY_normedToBell = (wave[i-1]-ave)*mult;
			float toY_normedToBell = (wave[i]-ave)*mult;
			int midHeight = height/2;
			int fromY = (int)(midHeight*(1+fromY_normedToBell));
			int toY = (int)(midHeight*(1+toY_normedToBell));
			//int fromY = (int)(startHeight+mult*wave[i-1]);
			//int toY = (int)(startHeight+mult*wave[i]);
			int fromX = width*(i-1)/wave.length;
			int toX = width*i/wave.length;
			g.drawLine(fromX, fromY, toX, toY);
		}
	}
	
	/** IN and CORRECT_OUT. */
	public DataSequence asDataSequence(){
		List<DataStep> list = new ArrayList();
		for(int i=0; i<data[0].length; i++){
			float in = data[IN][i];
			float correctOut = data[CORRECT_OUT][i];
			list.add(new DataStep(new float[]{in}, new float[]{correctOut}));
		}
		return new DataSequence(list);
	}

}
