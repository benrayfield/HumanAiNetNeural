/** Ben F Rayfield offers this software opensource MIT license */
package mutable.uitool;
import java.io.Closeable;
import java.util.function.Consumer;
import javax.swing.JPanel;

/** This is created normally by a string in listweb or typed in right column,
under which this JPanel appears. As Consumer, it gets that first string from one of the tabs
in the right column, and any updates it gets from the other tab, both of which
contain a textarea thats autofilled by listweb def but can be edited in the right column.
Calling Consumer.accept(Object) again is optional.
Consumer.accept(Object) is normally that String but later could be other object types
such as immutable.occamsfuncer.Funcer or JsonDS map.
An example of editing that is to change learningRate in RBM.
<br><br>
In that text, a uitool is (TODO replace javaclass:) starts as uitool:packagename.classname
then whitespace then the rest of the text, and (TODO) it sees that whole text
(TODO not just the part after that first token).
<br><br>
Subclasses must have a 0 param constructor.
*/
public abstract class Uitool extends JPanel implements Consumer, Closeable{
	public void accept(Object o){
		throw new Error("TODO override Consumer.accept");
	}
	
	public void close(){}
}
