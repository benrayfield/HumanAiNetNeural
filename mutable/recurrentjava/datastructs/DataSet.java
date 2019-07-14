package mutable.recurrentjava.datastructs;
import java.io.Serializable;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.model.Model;

public abstract class DataSet implements Serializable {
	public int inputDimension;
	public int outputDimension;
	public Loss lossTraining;
	public Loss lossReporting;
	public List<DataSequence> training;
	public List<DataSequence> validation;
	public List<DataSequence> testing;
	public abstract void DisplayReport(Model model, Random rng) throws Exception;
	public abstract Unaflop getModelOutputUnitToUse();
}
