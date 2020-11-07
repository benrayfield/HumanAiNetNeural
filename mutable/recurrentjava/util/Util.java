package mutable.recurrentjava.util;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import immutable.compilers.opencl_fixmeMoveSomePartsToImmutablePackage.FSyMem;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.MathUtil;
import mutable.recurrentjava.matrix.Matrix;

public class Util {
	
	public static int pickIndexFromRandomVector(Matrix probs, Random r) throws Exception {
		float mass = 1f;
		FSyMem probsW = probs.mem("w");
		for (int i = 0; i < probs.size; i++) {
			float prob = probsW.get(i) / mass;
			if(MathUtil.weightedCoinFlip(prob, r)){
			//if (r.nextFloat() < prob) {
				return i;
			}
			mass -= probsW.get(i);
		}
		throw new Exception("no target index selected");
	}
	
	public static float median(List<Float> vals) {
		Collections.sort(vals);
		int mid = vals.size()/2;
		if (vals.size() % 2 == 1) {
			return vals.get(mid);
		}
		else {
			return (vals.get(mid-1) + vals.get(mid)) / 2;
		}
	}

	public static String timeString(double milliseconds) {
		String result = "";
		
		int m = (int) milliseconds;
		
		int hours = 0;
		while (m >= 1000*60*60) {
			m -= 1000*60*60;
			hours++;
		}
		int minutes = 0;
		while (m >= 1000*60) {
			m -= 1000*60;
			minutes++;
		}
		if (hours > 0) {
			result += hours + " hours, ";
		}
		int seconds = 0;
		while (m >= 1000) {
			m -= 1000;
			seconds ++;
		}
		result += minutes + " minutes and ";
		result += seconds + " seconds.";
		return result;
	}
}
