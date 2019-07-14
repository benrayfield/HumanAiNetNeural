package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.*;

public class SimpleCx implements Cx{

	public long memFree(){
		throw new Todo();
	}

	public long memUsing(){
		throw new Todo();
	}

	public long memTotal(){
		throw new Todo();
	}

	public long compFree(){
		throw new Todo();
	}

	public boolean compUse(long use){
		throw new Todo();
	}

	public Zing get(Zing hash){
		throw new Todo();
	}

	public Zing ptr(Zing z){
		throw new Todo();
	}

	public Zing emptyMaplist(){
		throw new Todo();
	}

	public Zing emptyLeaf(){
		throw new Todo();
	}

	public Zing mapPutAll(Zing firstMaplist, Zing secondMaplist){
		throw new Todo();
	}

	public Zing mapPut(Zing maplist, Zing key, Zing val){
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

	public boolean econacyc(){
		throw new Todo();
	}

	public double econacycCost(Zing z, double time){
		throw new Todo();
	}

	public double econacycInPtrs(Zing z){
		throw new Todo();
	}

	public double econacycOutPtrs(Zing z){
		throw new Todo();
	}

	public double econacycRelValue(Zing z){
		throw new Todo();
	}

}
