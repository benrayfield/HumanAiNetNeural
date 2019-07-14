package mutable.listweb.ui;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Consumer;

import javax.swing.ListModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import mutable.listweb.ListwebRoot;

/** As Consumer<String>, listens for changes to nameOfParentOfPrilist's value
which includes nameOfParentOfPrilist.prilist.
*/
public class WrapPrilistInListModel implements ListModel, Consumer<String>{

	public final String nameOfParentOfPrilist;
	
	protected final Set<ListDataListener> listDataListeners = new HashSet();
	
	public WrapPrilistInListModel(String nameOfParentOfPrilist){
		this.nameOfParentOfPrilist = nameOfParentOfPrilist;
		ListwebRoot.startListening(this, nameOfParentOfPrilist);
	}

	public int getSize(){
		return ListwebRoot.prilist(nameOfParentOfPrilist).size();
	}

	public Object getElementAt(int i){
		return ListwebRoot.prilist(nameOfParentOfPrilist).get(i);
	}

	public void addListDataListener(ListDataListener x){
		listDataListeners.add(x);
	}

	public void removeListDataListener(ListDataListener x){
		listDataListeners.remove(x);
	}

	public void accept(String eventAboutThisName){
		if(eventAboutThisName.equals(nameOfParentOfPrilist)){
			ListDataEvent e = new ListDataEvent("unknownWhichObjectItCameFrom",
				ListDataEvent.CONTENTS_CHANGED, 0, getSize()-1);
			for(ListDataListener x : listDataListeners){
				x.contentsChanged(e);
			}
		}
	}

}
