/** Ben F Rayfield offers this software opensource MIT license */
package mutable.listweb;

public class Debug{
	private Debug(){}
	
	public static final boolean
		logGetTodoTime = false,
		logSwingLock = false,
		logJListEvents = false,
		logSwingInvoke = false,
		logListwebEvents = true,
		logDroppingOfPossiblyInfinitelyLoopingAcycEvents = true,
		logSetScroll = true,
		logPrilistScrollFractionFromMap = false,
		logStartsAndStopsOfListening = true,
		logSetOfListenersBeforeFiringEvent = false,
		logModified = false,
		skipRootCharsUpdateOnBootForSpeed = true; //do it through Action menu

}