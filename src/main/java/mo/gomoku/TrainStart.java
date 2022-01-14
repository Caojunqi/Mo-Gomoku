package mo.gomoku;

import ai.djl.engine.Engine;
import mo.gomoku.mcts.MctsTrainer;

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
