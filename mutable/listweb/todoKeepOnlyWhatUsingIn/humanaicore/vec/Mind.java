/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec;
import java.util.Arrays;
import java.util.Collection;
import java.util.function.Consumer;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.askmut.AskMutable;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.old.MutZing;

/** A stateful or stateless object which reads andOr writes a parameter Zing.
My LongSize matches param Zing's LongSize, but I may have far bigger state
such as List<List<WeightsNode>> or another Zing.
<br><br>
think<Zing> or maybe as Consumer<Zing>: z.size()==size()
<br><br>
IMPORTANT: Mind and Statsys are NOT Consumer<Zing> because the func
whose param is Zing must return a Mind or Statsys
whose state has changed (such as Statsys learned).
If its mutable, returns itself/this.
If immutable, returns a different object.
*/
public interface Mind<T extends Mind<T>> extends LongSize, AskMutable, AskChanceSymmetry{
	
	/** Returns a Mind which may have learned (or this object was changed in some way) from z,
	or if is no change (FIXME TODO are immutable() and stateless() different that way?) returns this.
	z.size()==size().
	*/
	public T think(MutZing z);
	default public T thinks(MutZing... zs){
		return thinks(Arrays.asList(zs));
	}
	default public T thinks(Collection<MutZing> zs){
		T t = (T) this;
		for(MutZing z : zs) t = t.think(z);
		return (T) t;
	}
	
	/** Immutable ptr and contents */
	public MutZing reads();
	
	/** Immutable ptr and contents */
	public MutZing writes();
	
	//TODO I want to keep Mind simple, so is it excessive to extend ForkMutable<Mind>?
	//Probably so. Mind is supposed to be the smallest set of things you need
	//to readAndOr write a parameter Zing. And it is.
	//So proceed to rbm scalar low bits higher mousemoveai.
	
	//"TODO be a java8 Consumer<Zing> instead of think func? And what about the parallel thinking of multiple Zing like in the older Statsys (especially in rbm annealing all at once)? Is Spliterator etc useful for that? Or is it usually not splittable (like that double annealing loop)?"
	
	/** If mutable(), returns this same Statsys. Else a Statsys modified by having learned those. *
	public Statsys think(Collection<FuzzyVec> io);
	/** Same as think(Collection<Statsys>) *
	public Statsys think(FuzzyVec... io);
	*/
	
	/** True if is a wrapper for a simulated circuit like the bits of plus, multiply,
	a large memory thats readable andOr writable, or anything other than AI.
	Can use float64 as ptr or bit. Can even use 8 float64s as ptr at acycPair,
	(TODO) but probably Zing is better for that (which is obsoleting this and related interfaces).
	*
	public boolean isCodeWrapper();
	*/
	
	//"TODO Instead of FuzzyVec's weight() Vec, there should be 2 Vecs: in and out, and maybe float64 means the chance it reads andOr writes or how much it reads andOr writes. Its important to differentiate between chance it writes vs it writes often but changes it only a little so maybe that should be another Vec but do we need that? Or simpler, like JSoundCard's SoundFunc, use ints and always have the ints of how many are read and how many are written and if big enough they overlap on some indexs."
	
	//"TODO Statsys is my subtype which does predict andOr learn, and maybe think has multiple of them"

}