package mutable.listweb.todoKeepOnlyWhatUsingIn.zing.langs;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Text;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.*;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.*;

/** translates between utf8 string and zing of those content bits */
public class StringLang implements Lang<String>{

	public String o(Zing z){
		if((z.sizeBits()%8) != 0) throw new Err("Not a utf8 string cuz not a multiple of 8 bits");
		return Text.bytesToString(ZingRoot.contentBytesFromLeaf(z));
	}

	public Zing z(String o){
		return ZingRoot.leafFromContentBytes(Text.stringToBytes(o));
	}
	
	

}
