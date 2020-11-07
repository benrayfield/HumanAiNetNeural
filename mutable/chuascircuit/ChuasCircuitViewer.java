package mutable.chuascircuit;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

import javax.swing.JPanel;

import org.lwjgl.opengl.Display;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.util.Time;
import mutable.util.ui.ScreenUtil;

/**
*/
public class ChuasCircuitViewer extends JPanel{
	
	/*TODO the 2 kinds of viewing the testcase described in ChuasCircuitLearnedByRecurrentJava,
	one where its using a perfect ChuasCircuit to generate the inputs
	and the other where its all done by LSTM,
	and in both cases inputs are delayed by a given number of cycles
	so somewhere the LSTM's outputs will have to be remembered
	and use the LSTM's outputs from delayCycles (offby1?) back
	divided by dtPerAICycle (a constant in ChuasCircuitLearnedByRecurrentJava constructor)
	add that to x y and z. Then Display. Given a time range of a perfect ChuasCircuit
	a range of that delay, the LSTM uses that for the short delay to boot
	then continues estimating chuascircuit by pure LSTM.
	*/
	
	public final ChuasCircuit[] c;
	
	/** sim speed is this many times faster/slower than UTC time */
	protected double timeMult = .2;
	
	public ChuasCircuitViewer(ChuasCircuit... c){
		this.c = c;
		new Thread(()->{ while(true) { ChuasCircuitViewer.this.repaint(); Time.sleepNoThrow(.01); } }).start();
		addMouseMotionListener(new MouseMotionListener(){
			public void mouseMoved(MouseEvent e){
				for(ChuasCircuit cc : c){
					cc.x += .01;
				}
			}
			public void mouseDragged(MouseEvent e){
				throw new Error("TODO");
			}
		});
	}
	
	double timeLastPaint = Time.now();
	
	public void paint(Graphics g){
		g.setColor(Color.black);
		g.fillRect(0, 0, getWidth(), getHeight());
		double now = Time.now();
		double dt = now-timeLastPaint;
		dt = MathUtil.holdInRange(0, dt, .1);
		//dt *= .1;
		int timeSteps = 10;
		dt /= timeSteps;
		dt *= timeMult;
		for(int timeStep=0; timeStep<timeSteps; timeStep++){
			int i = 0;
			for(ChuasCircuit cc : c){
				cc.nextState(dt);
				g.setColor(((i&1)==0) ? new Color(0f,.55f,0f) : new Color(.05f,.05f,1f));
				double magnify = 100;
				int x = ((int)(cc.x*magnify))+getWidth()/2;
				int y = ((int)(cc.y*magnify))+getHeight()/2;
				//TODO z
				//double r = 2+30*MathUtil.sigmoid(c.z*3.1);
				double r = 2+30*MathUtil.sigmoid(cc.z*.4);
				g.fillOval(x-(int)r, y-(int)r, (int)(2*r), (int)(2*r));
				i++;
			}
		}
	}
	
	public static void main(String[] args){
		/*ChuasCircuit c[] = {
			new ChuasCircuit(),
			new ChuasCircuitLearnedByRecurrentjava(.0001)
		};*/
		//ChuasCircuit c[] = new ChuasCircuit[30];
		//ChuasCircuit c[] = new ChuasCircuit[1];c
		ChuasCircuit c[] = new ChuasCircuit[100];
		for(int i=0; i<c.length; i++){
			c[i] = ChuasCircuitLearnedByRecurrentjava.randomChuasCircuitStartState(Rand.strongRand);
		}
		/*for(ChuasCircuit cc : c){
			cc.x = .7;
			cc.y = 0;
			cc.z = 0;
			cc.c1 = 15.6;
			cc.c2 = 1;
			cc.c3 = 28;
			cc.m0 = -1.143;
			cc.m1 = -0.714;
		}*/
		ScreenUtil.testDisplayWithExitOnClose(new ChuasCircuitViewer(c));
	}

}
