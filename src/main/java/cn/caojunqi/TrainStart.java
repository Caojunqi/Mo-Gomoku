package cn.caojunqi;

import ai.djl.engine.Engine;
import cn.caojunqi.mcts.MctsTrainer;

/**
 * 训练启动类
 *
 * @author Caojunqi
 * @date 2022-01-11 21:05
 */
public class TrainStart {

	public static void main(String[] args) {
		Engine.getInstance().setRandomSeed(0);
		MctsTrainer mctsTrainer = new MctsTrainer();
		mctsTrainer.run();
	}
}
