package mutable.listweb;

public class Debug{
	private Debug(){}
	
	public static final boolean
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