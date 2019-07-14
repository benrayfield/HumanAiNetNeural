package immutable.rbm.learnloop.slideactions;
import immutable.rbm.learnloop.LearningVec;
import immutable.rbm.learnloop.SlideAction;

public class AddLearningVec implements SlideAction{
	
	public final LearningVec lv;
	
	public AddLearningVec(LearningVec lv){
		this.lv = lv;
	}

}
