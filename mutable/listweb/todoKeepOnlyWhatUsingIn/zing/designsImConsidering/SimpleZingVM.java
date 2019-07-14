package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.designsImConsidering;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Zing;

public class SimpleZingVM implements ZingVM{
	
	//"TODO must computeMoney and memoryMoney be synchronized? Depends how accurate they must be. If synchronized, may slow things too much since this would be called often, but what if it was only used within the same thread, as ZingRoot.threadState is already designed, and only synchronize when communicating between those? No Zing can contain computeMoney or memoryMoney since zing is immutable."

	public long m(){
		throw new Todo();
	}

	public long c(){
		throw new Todo();
	}

	public boolean c(long use){
		throw new Todo();
	}

	public ZingVM take(long memoryMoney, long computeMoney){
		throw new Todo();
	}

	public Zing empty(){
		throw new Todo();
	}

	public Zing mapPutAll(Zing firstMap, Zing secondMap){
		throw new Todo();
	}

	public Zing listCat(Zing firstList, Zing secondList){
		throw new Todo();
	}

	public Zing leafCat(Zing firstLeaf, Zing secondLeaf){
		throw new Todo();
	}

	public Zing funcall(Zing func, Zing param){
		throw new Todo();
	}

	public long memFree() {
		throw new Todo();
	}

	public long memUsing() {
		throw new Todo();
	}

	public long memTotal() {
		throw new Todo();
	}

}
