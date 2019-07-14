package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.hash.Sha256;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec.impl.ConstBitstring;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.Zing;
import immutable.occamsjsonds.JsonDS;

/** equality is by == of ob, and hashcode a function of System.identityHashCode of ob.
In the global network, equality is by the string hash which is by default merkle sha256 of json.
*/
@Deprecated //Zing will do this internally, lazyEvaled using its hash()
public class Node<T>{
	
	/*TODO?
	public double econacycCost;
	
	public float econacycRelValue;
	
	public long incomingPointers;
	*/
	
	/** immutable json object: NavigableMap, List, String, Double, or Zing */
	public final T ob;
	
	/** If sha2, such as sha256 or sha512, starts with h$ then base64 of that hashcode padded with bit1 then 0-5 bit0s.
	If sha3, starts with H$. If string literal, starts with g$.
	*/
	private String hash;
	
	public Node(T ob){
		this.ob = ob;
	}
	
	public String hash(){
		if(hash == null){
			String j = json();
			byte b[] = Text.stringToBytes(j);
			byte sha[] = Sha256.sha256(b);
			return "h$"+Base64.fromBits(new ConstBitstring(sha));
		}
		return hash;
	}
	
	/** If ob is a Zing, returns string of hash since Zing is a bitstring and must only be used by hash at the json level.
	The hash includes a prefix that means its a hash, such as h$ for sha2 or H$ for sha3. g$ is prefix of string literal.
	*/
	public String json(){
		return JsonDS.jsonString(this);
		//"TODO ZingJson.toJson(ob) except use hash (with h$ prefix if sha2) instead of expanding"
	}
	
	public boolean isSaved(){
		throw new Todo();
	}
	
	/** recursively save to longterm storage (such as harddrive or internet) */
	public void save(){
		if(ob instanceof Collection){
			Object o = ob;
			if(o instanceof Map) o = (T) ((Map)o).values();
			for(Object child : (Collection)o){
				if(child instanceof NavigableMap || child instanceof List || child instanceof Zing){
					throw new Todo("TODO save child, which may be NavigableMap, List, ");
				}else{
					throw new Todo("TODO what about big string, are those saved separately?");
				}
			}
		}
	}
	
	public int hashCode(){
		return System.identityHashCode(ob)+237732523;
	}
	
	public boolean equals(Object o){
		if(!(o instanceof Node)) return false;
		return ob == ((Node)o).ob;
	}

}