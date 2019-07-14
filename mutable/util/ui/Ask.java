/** Ben F Rayfield offers this software opensource MIT license */
package mutable.util.ui;
import static mutable.util.Lg.*;
import javax.swing.JOptionPane;

public class Ask{
	
	public static void ok(String message){
		JOptionPane.showMessageDialog(null, message);
	}
	
	public static boolean yesNo(String yesNoQuestion){
		int input = JOptionPane.showConfirmDialog(null, yesNoQuestion);
        // 0=yes, 1=no, 2=cancel
		return input==0;
		//TODO remove the Cancel button. Should only be Yes and No.
	}
	
	public static void main(String[] args){
		ok("test");
	}

}