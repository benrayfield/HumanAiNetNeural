/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.vec;


public interface BitVsScalar{
	
	public boolean preferScalarInsteadBit();
	
	/** True if prefer fraction instead of inverseSigmoid.
	d funcs are fraction. ds funcs are inverseSigmoid of that fraction.
	All Zings work with all of those and bits but differ in precision andOr roundoff.
	*
	public boolean preferFractionInsteadInverseSigmoid();
	*/

	public boolean precisionFloat64();

}
