/** Ben F Rayfield offers this software opensource MIT license */
package mutable.util.ui;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.swing.JPanel;

import mutable.util.Rand;

/** Paints a column 1 pixel wide using a ColorAtFraction,
then slide the left and right halfs 1 pixel away from center.
*/
public class ColorFlowFromCenterColumn extends JPanel implements MouseMotionListener, MouseListener, KeyListener{
	
	protected ColorAtFraction painter;
	
	public Map params;
	
	protected long paints;
	
	protected int pixelsSlideOverPerPaint;
	
	public ColorFlowFromCenterColumn(ColorAtFraction firstPainter, Map firstParams, int pixelsSlideOverPerPaint){
		this.params = firstParams;
		this.pixelsSlideOverPerPaint = pixelsSlideOverPerPaint;
		addMouseMotionListener(this);
		addMouseListener(this);
		setFocusable(true); //for KeyListener
		addKeyListener(this);
		setPainter(firstPainter);
	}
	
	public void setPainter(ColorAtFraction c){
		painter = c;
	}
	
	public void paint(Graphics g){
		int w = getWidth(), h = getHeight();
		g.setPaintMode();
		if(paints++ < 2){ //TODO why is fillRect ignored on the first call of paint?
			g.setColor(Color.black);
			g.fillRect(0, 0, w, h);
		}
		final int column = w/2;
		g.copyArea(pixelsSlideOverPerPaint, 0, column, h, -pixelsSlideOverPerPaint, 0);
		g.copyArea(column-pixelsSlideOverPerPaint, 0, w-column, h, pixelsSlideOverPerPaint, 0);
		for(int y=0; y<h; y++){
			int color = painter.colorAtFraction((y+.5)/h, params);
			g.setColor(new Color(color));
			g.fillRect(column-pixelsSlideOverPerPaint, y, pixelsSlideOverPerPaint*2+1, 1);
		}
	}

	public void mouseDragged(MouseEvent e){ mouseMoved(e); }

	public void mouseMoved(MouseEvent e){
		int h = getHeight(), w = getWidth(), x = e.getX(), y = e.getY();
		params.put("heightPixels", (double)h);
		params.put("widthPixels", (double)w);
		params.put("mouseXPixels", (double)x);
		params.put("mouseYPixels", (double)y);
		params.put("mouseXFraction", (double)Math.max(0, Math.min(x/(w-1.), 1)));
		params.put("mouseYFraction", (double)Math.max(0, Math.min(y/(h-1.), 1)));
		params.put("mouseIn", 1.);
	}

	public void mouseClicked(MouseEvent e){}

	public void mousePressed(MouseEvent e){
		params.put("mouseButton"+e.getButton(),true);
	}

	public void mouseReleased(MouseEvent e){
		params.remove("mouseButton"+e.getButton());
	}

	public void mouseEntered(MouseEvent e){
		if(params.containsKey("mouseYFraction")){
			params.put("mouseIn", 1.);
		}
	}

	public void mouseExited(MouseEvent e){
		params.put("mouseIn", 0.);
	}

	public void keyTyped(KeyEvent e){}

	public void keyPressed(KeyEvent e){
		//String t = KeyEvent.getKeyText(e.getKeyCode());
		//System.out.println(t);
		//params.put(t,true);
		//System.out.println(e.getKeyChar());
		params.put((double)e.getKeyChar(),true);
	}

	public void keyReleased(KeyEvent e){
		//params.remove(KeyEvent.getKeyText(e.getKeyCode()));
		params.remove((double)e.getKeyChar());
	}

}

