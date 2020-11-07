package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ui;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import mutable.util.ui.ScreenUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl.ConstBitstring;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl.ImmutScaVec;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Zing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ZingRoot;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl.LongArrayZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;

/** paint video 1 frame at a time as a Zing.
TODO as a 

Can paint magified using Graphics.fillRectangle or per pixel as BufferedImage.
If the Zing is mutable, just call repaint(). If immutable (as zingjson always creates them), use the other repaint func.
Each int32 in 32 bit blocks in the Zing is ARGB.
*/
public class ZingVideo extends JPanel{
	
	protected Zing z;
	protected int zWidth = 1;
	//protected float zMagY=1, zMagX=1, zLeft, zTop;
	protected AffineTransform zAftrans;
	
	public ZingVideo(){
		setMinimumSize(new Dimension(5000,5000));
		setPreferredSize(new Dimension(5000,5000));
	}
	
	/** Painted after updating the Graphics object with the AffineTransform, including for magnify. 
	OLD: null unless zMagY and zMagX are both small, and is replaced when size changes
	*/
	protected BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_4BYTE_ABGR);
	
	public void paint(Graphics gr){
		if(z == null){
			gr.setColor(Color.black);
			gr.fillRect(0, 0, getWidth(), getHeight());
		}else{
			if(gr instanceof Graphics2D){
				Graphics2D g = (Graphics2D)gr;
				//lg("colormodel: "+img.getColorModel().getClass());
				//if(!zAftrans.isIdentity()) g.setTransform(zAftrans);
				
				//g.setColor(Color.green);
				//g.drawString("test string", 0, 0);
				
				//replace BufferedImage if size is different
				//then paint the ints into BufferedImage
				//then paint it onto Graphics
				//or more directly, consider an ImageProducer wrapper of Zing
				int offset = z.isBig() ? 4 : 1;
				int zSizeInInts = (int)(z.sizeBits()>>>5)-offset;
				int zHeight = (zSizeInInts+zWidth-1)/zWidth;
				//int pixels = zWidth*zHeight;
				if(img.getHeight() != zHeight || img.getWidth() != zWidth){ //replace BufferedImage
					img = new BufferedImage(zWidth, zHeight, BufferedImage.TYPE_4BYTE_ABGR);
				}
				for(int y=0; y<zHeight; y++){
					for(int x=0; x<zWidth; x++){
						int color = z.ia(offset+x);
						img.setRGB(x, y, color);
					}
					offset += zWidth;
				}
				g.drawImage(img, zAftrans, this);
			}else{
				throw new Err("Unknown graphics type: "+gr.getClass().getName());
			}
			
			/*int minMagForFillRect = 4; //draw it the more efficient way.
			//TODO take statistics in realtime on what size magnify BufferedImage vs fillRect is faster,
			//not counting alloc of BufferedImage.
			if(zMagY < minMagForFillRect && zMagX < minMagForFillRect){ //BufferedImage
				//replace BufferedImage if size is different
				//then paint the ints into BufferedImage
				//then paint it onto Graphics
				//or more directly, consider an ImageProducer wrapper of Zing
				throw new Todo();
			}else{ //fillRect
				TODO
			}*/
		}
	}
	    		 
	    		 
	public void repaint(Zing pixels, int width, AffineTransform aftrans){
		this.z = pixels;
		this.zAftrans = aftrans;
		this.zWidth = width;
		repaint();
	}
	
	/** If left is negative, part of the image hangs off the left side of the screen. Similar for top. *
	public void repaint(Zing pixels, int width, float magnifyY, float magnifyX, float left, float top){
		z = pixels;
		zWidth = width;
		zMagY = magnifyY;
		zMagX = magnifyX;
		zLeft = left;
		zTop = top;
		repaint();
	}*/
	
	public static void testDisplay(ZingVideo z){
		double m00=16, m01=7, m02=200;
		double m10=-7, m11=16, m12=220;
		AffineTransform aftrans = new AffineTransform(m00, m10, m01, m11, m02, m12);
		//z.repaint(ConstBitstring.EMPTY, 100, aftrans);
		long g[] = new long[200];
		for(int i=0; i<g.length; i++){
			int colorA = 0xff000000 | Rand.strongRand.nextInt(0x1000000);
			int colorB = 0xffabcdef;
			g[i] = (((long)colorA)<<32) | (colorB&0xffffffffL);;
		}
		//z.repaint(new ConstBitstring(g), 10, aftrans);
		//z.repaint(new LongArrayZing(g), 10, aftrans);
		z.repaint(ZingRoot.bitstring(g), 20, aftrans);
	}
	
	public static void main(String[] args){
		JFrame window = new JFrame("test "+ZingVideo.class.getName());
		ZingVideo z = new ZingVideo();
		window.add(z);
		window.setSize(300, 300);
		ScreenUtil.moveToScreenCenter(window);
		window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		testDisplay(z);
		window.setVisible(true);
	}

}