package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.ui;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.*;

import java.awt.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;

import mutable.listweb.ListwebRoot;
import mutable.listweb.ListwebUtil;
import mutable.util.ui.ScreenUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;

public class ZingWindow extends JFrame{
	
	public ZingWindow(String windowTitle, String upStackName, String downStackName, String selectPtrName,
			String nameOfSearchResults, boolean exitOnClose){
		super(windowTitle);
		add(new ZingPanel(upStackName, downStackName, selectPtrName, nameOfSearchResults));
		final boolean screenEnabled[] = {false};
		final JFrame window = this;
		int defaultWidth = 400;
		Runnable toggleScreen = ()->{
			screenEnabled[0] = !screenEnabled[0];
			ZingPanel z = (ZingPanel) window.getContentPane().getComponent(0);
			Dimension ss = Toolkit.getDefaultToolkit().getScreenSize();
			z.enableScreen(screenEnabled[0]);
			if(screenEnabled[0]){
				//z.setResizeWeight(.3);
				window.setSize(new Dimension((int)ss.getWidth()*2/3, (int)window.getBounds().getHeight()));
			}else{
				window.setSize(new Dimension(defaultWidth, (int)window.getBounds().getHeight()));
				//z.setResizeWeight(0);
			}
			ScreenUtil.moveToScreenCenter(window);
		};
		setJMenuBar(ListwebUtil.newListwebMenuBar(selectPtrName, upStackName, downStackName, toggleScreen));
		byte iconBytes[] = mutable.util.Files.readFileOrInternalRel("/data/humanaicore/icon.jpg");
		try{
			setIconImage(ImageIO.read(new ByteArrayInputStream(iconBytes)));
		}catch(IOException e){ throw new Err(e); }
		if(exitOnClose){
			addWindowListener(new WindowListener(){
				public void windowOpened(WindowEvent e){}
				public void windowClosing(WindowEvent e){
					try{
						lg("Window is closing...");
						ListwebRoot.onClosingProg();
					}finally{
						System.exit(0);
					}
				}
				public void windowClosed(WindowEvent e){}
				public void windowIconified(WindowEvent e){}
				public void windowDeiconified(WindowEvent e){}
				public void windowActivated(WindowEvent e){}
				public void windowDeactivated(WindowEvent e){}
			});
		}
		Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		//window.setSize(screen.width/4, screen.height-150);
		//window.setSize(screen.width/4, screen.height-50);
		//setSize(1000,700);
		setSize(defaultWidth,700);
		ScreenUtil.moveToScreenCenter(this);
		setLocation((screen.width-this.getWidth())/2, 10);
		setVisible(true);
		System.out.println("Window setVisible "+this);
		System.out.println("...");
	}

}
