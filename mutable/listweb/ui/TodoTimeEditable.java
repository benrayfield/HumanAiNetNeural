package mutable.listweb.ui;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.function.Consumer;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import mutable.listweb.ListwebRoot;
import mutable.listweb.ListwebUtil;
import mutable.listweb.Options;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;

/** listens (Consumer<String>) for changes to aSelectPtr.selectedName
and displays its "todoTime" field editable. Its a float64 utc time but
is editable as gregorian time with buttons to change it slow or fast or set it to now.
This is normally used above a DefPanel and above its NameFieldNoneditable.
*/
public class TodoTimeEditable extends JPanel implements Consumer<String>{
	
	public final String selectPtr;
	
	protected String prevOrCurrentSelected;
	
	public final JLabel text = new JLabel();
	
	public TodoTimeEditable(String selectPtr){
		super(new BorderLayout());
		//setMinimumSize(new Dimension(5,0));
		//setPreferredSize(new Dimension(100,0));
		this.selectPtr = selectPtr;
		ListwebRoot.startListening(this, selectPtr);
		prevOrCurrentSelected = (String) ListwebRoot.get(selectPtr).get("selectedName");
		if(prevOrCurrentSelected != null){
			ListwebRoot.startListening(this, prevOrCurrentSelected);
		}
		JPanel left = new JPanel(new GridLayout(1,0));
		left.add(button("Y",365*24*60*60)); //TODO also a button to set to a small random time before all other times, making it the next thing to do
		left.add(button("M",30*24*60*60));
		left.add(button("W",7*24*60*60));
		left.add(button("D",24*60*60));
		left.add(button("H",60*60));
		left.add(button("5",5*60));
		add(left, BorderLayout.WEST);
		add(text, BorderLayout.CENTER);
	}
	
	protected JButton button(String label, final double dt){
		JButton b = new JButton(new AbstractAction(label){
			public void actionPerformed(ActionEvent e){
				System.out.println(e.getSource());
				addTime(dt);
			}
		});
		b.setMargin(new java.awt.Insets(0,0,0,0));
		return b;
	}
	
	public void addTime(double dt){
		String selectedName = (String) ListwebRoot.get(selectPtr).get("selectedName");
		if(selectedName != null){
			ListwebRoot.setTodoTime(selectedName, ListwebRoot.getTodoTime(selectedName)+dt);
		}
	}
	
	static boolean equals(Object a, Object b){
		return a!=null ? a.equals(b) : b==null;
	}
	
	/** fraction is distance across displayed monospaced timeToString(double). Returns NaN if should remove the time. */
	public static double uiSecondsIncrementPerClickAtHorizontalFraction(double fraction){
		double size = 18; //"XYY.MM.DDW.HHP.MM ".length();
		if(fraction < 1/size) return 0./0; //NaN means remove
		if(fraction < 3/size) return 365*24*60*60; //YY
		if(fraction < 6/size) return 30*24*60*60; //.MM
		if(fraction < 9/size) return 24*60*60; //.DD
		if(fraction < 11/size) return 7*24*60*60; //W.
		if(fraction < 14/size) return 60*60; //HHP
		return 5*60; //MM by 5 minute intervals
	}
	
	/** used if Options.option(ListwebUtil.optionUseTodotimesAndHdwmyColors, false) is true */
	public static final String timeToStringDisabled = timeToString(0.).replace('.','x');
	
	public static String timeToString(double time){
		if(time == 0) return ".................";
		Calendar c = Calendar.getInstance();
		c.setTimeInMillis((long)(time*1000));
		return "X"+ //X is button to remove the time
			(c.get(c.YEAR)%100)+"."
			+atLeastTwoDigits(c.get(c.MONTH)+1)+/*"JaFeMaApMaJnJyAuSeOcNoDe".charAt(c.get(c.MONTH))+*/"."
			+atLeastTwoDigits(c.get(c.DAY_OF_MONTH))+"UMTWHFS".charAt(c.get(c.DAY_OF_WEEK)-1)+"."
			+atLeastTwoDigits(c.get(c.HOUR)==0?12:c.get(c.HOUR))+(c.get(c.AM_PM)==c.AM?'A':'P')+"."
			+atLeastTwoDigits(c.get(c.MINUTE));
	}
	
	static String atLeastTwoDigits(int i){
		String s = ""+i;
		return s.length()==1 ? "0"+s : s;
	}
	
	/*UPDATE: I want every mindmap name displayed anywhere to be prefixed by something
	like 2017-09-14H-7P-45 (or 'XX-W-XX-XXxxx-XX if not scheduled), and left click adds to it,
	and right click subtracts, and clicking the 14 goes up or down 1 day, and clicking
	the Thu goes up or down 1 week, and days of week are MTWHFSU. This therefore means
	I can see and adjust time without going down to the def panel. The event system
	should handle this. Will take a little more work in the EditPrilist and related
	code to display it that way. Try to do it without putting the time in the list.
	*/

	public void accept(String aSelectPtr){
		String newSelected = (String) ListwebRoot.get(selectPtr).get("selectedName");
		if(!equals(newSelected,prevOrCurrentSelected)){
			if(prevOrCurrentSelected != null && !prevOrCurrentSelected.equals(selectPtr)){
				ListwebRoot.stopListening(this, prevOrCurrentSelected);
			}
			prevOrCurrentSelected = newSelected;
			if(prevOrCurrentSelected != null){
				ListwebRoot.startListening(this, prevOrCurrentSelected);
			}
		}
		String selectedName = (String) ListwebRoot.get(selectPtr).get("selectedName");
		if(selectedName == null){
			text.setText("");
		}else{
			double todoTime = ListwebRoot.getTodoTime(selectedName);
			//String s = new SimpleDateFormat("yy.M.dEEE-Ka-mm").format(new Date((long)(todoTime*1000)));
			//Date d = new Date((long)(todoTime*1000));
			String s = timeToString(todoTime);
			
			//text.setText(s+" TODO buttons to edit time, and if its 0 then display empty. Also adjust search to sort everything with a todoTime first, and within those 2 groups (with and without a todoTime, sort by shortest name)");
			text.setText(s+" todo make the parts of the time left and right clickable, but make it visually obvious or at least a tooltip. todo modify search.");
		}
		invalidate();
	}

}
