/** Ben F Rayfield offers this software opensource MIT license */
package mutable.learnloop_partsMovedToMutableuitoollearnloop.old.ui;
import static mutable.util.Lg.*;
import java.awt.Color;
import java.awt.Component;
import java.util.NavigableMap;
import java.util.function.Consumer;

import javax.swing.JTextArea;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import immutable.rbm.learnloop.LearnLoop;
import immutable.rbm.learnloop.RBM;
import mutable.compilers.java.JavaCompilers;
import mutable.learnloop_partsMovedToMutableuitoollearnloop.old.func.LearnLoopParam_OLD;
import mutable.util.Var;

public class LearnFuncRbmEditor extends JTextArea{
	
	//TODO use Learnloop.parseLearnloopScript (and move that to LearnloopUtil?)
	
	/** TODO merge the Vars when binufnode */
	public final Var<RBM> varRbm;
	
	/** TODO merge the Vars when binufnode. Var<NavigableMap> is forkEdited by JsonDS. */
	public final Var<NavigableMap> params;
	
	protected final Consumer<Var> listener;
	
	protected boolean ignoreOnChange;
	
	public LearnFuncRbmEditor(Var<RBM> varRbm, Var<NavigableMap> params){
		this.varRbm = varRbm;
		this.params = params;
		setText(varRbm.get().learnFunc);
		varRbm.startListening(listener = (Var var)->{
			/*TODO in case something externally changes the RBM.learnFunc (such as quickloading another RBM using number buttons)
			but this is causing ui problem
			ignoreOnChange = true;
			String code = varRbm.get().learnFunc;
			if(!code.equals(getText())){
				setText(code);
			}
			ignoreOnChange = false;
			*/
		});
		getDocument().addDocumentListener(new DocumentListener(){
			public void insertUpdate(DocumentEvent e){
				LearnFuncRbmEditor.this.onChange();
			}
			public void removeUpdate(DocumentEvent e){
				LearnFuncRbmEditor.this.onChange();
			}

			public void changedUpdate(DocumentEvent e){
				LearnFuncRbmEditor.this.onChange();
			}
			
		});
	}
	
	protected void finalize() throws Throwable{
		varRbm.stopListening(listener);
	}
	
	protected void onChange(){
		if(ignoreOnChange) return;
		String editingCode = getText();
		lg("learnFunc textarea: "+editingCode);
		String code = varRbm.get().learnFunc;
		Color pass = Color.WHITE, fail = new Color(.8f, .8f, .8f);
		if(!code.equals(editingCode)){
			try{
				//LearnLoopParam_OLD.compileSandboxed(editingCode);
				LearnLoop.compileSandboxed(editingCode, JavaCompilers.get(false, false));
				lg("learnFunc textarea COMPILED");
				setBackground(pass);
				varRbm.set(varRbm.get().setLearnFunc(editingCode));
			}catch(Throwable t){
				lg("learnFunc textarea FAIL (on CODE: "+code+") cuz "+t);
				setBackground(fail);
				//lgErr(t);
			}
		}else{
			lg("learnFunc textarea NO CHANGE.");
			setBackground(pass);
		}
		lg("learnFunc textarea END");
	}

}
