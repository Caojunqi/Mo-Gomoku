package mo.gomoku.mcts;

import ai.djl.Model;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import mo.gomoku.game.Board;
import mo.gomoku.util.ModelBuilder;
import org.apache.commons.lang3.Validate;

/**
 * 五子棋模型测试类
 *
 * @author Caojunqi
 * @date 2022-01-13 11:59
 */
public class MctsTester {
	/**
	 * 游戏环境
	 */
	private Board gameEnv;
	private int alphaPlayerId;
	/**
	 * 对手智能体
	 */
	private MctsAlphaAgent alphaAgent;
	/**
	 * 对战是否开始
	 */
	private boolean start;


	public MctsTester(Board gameEnv, boolean alphaFirst) {
		this.gameEnv = gameEnv;
		this.alphaPlayerId = alphaFirst ? 0 : 1;

		setupOpponents();
	}

	/**
	 * 战斗开始
	 */
	public void start() {
		if (this.start) {
			// 对战已经开始
			return;
		}
		gameEnv.reset();
		this.start = true;
		int curPlayerId = this.gameEnv.getCurPlayerId();
		while (curPlayerId == this.alphaPlayerId) {
			// 机器人行动
			int robotAction = this.alphaAgent.getAction(this.gameEnv);
			this.gameEnv.doMove(robotAction);
			curPlayerId = this.gameEnv.getCurPlayerId();
		}
	}

	/**
	 * 玩家行动
	 */
	public void playerAction(int action) {
		if (!this.start) {
			// 对战尚未开始
			return;
		}
		int curPlayerId = this.gameEnv.getCurPlayerId();
		if (curPlayerId == this.alphaPlayerId) {
			// 尚未到玩家行动
			return;
		}
		this.gameEnv.doMove(action);
		if (this.gameEnv.checkGameOver().first) {
			this.start = false;
			return;
		}
		curPlayerId = this.gameEnv.getCurPlayerId();
		while (curPlayerId == this.alphaPlayerId) {
			// 机器人行动
			int robotAction = this.alphaAgent.getAction(this.gameEnv);
			this.gameEnv.doMove(robotAction);
			curPlayerId = this.gameEnv.getCurPlayerId();
		}
		if (this.gameEnv.checkGameOver().first) {
			this.start = false;
		}
	}

	/**
	 * 构建对手
	 */
	private void setupOpponents() {
		Model opponentModel = ModelBuilder.buildPretrainedModel(MctsParameter.BEST_MODEL_PREFIX);
		Validate.notNull(opponentModel, "没有训练好的模型，不能进行性能测试");
		TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
		Trainer trainer = opponentModel.newTrainer(config);
		trainer.initialize(this.gameEnv.getStateShape());
		trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
		this.alphaAgent = new MctsAlphaAgent(trainer);
	}

	public Board getGameEnv() {
		return gameEnv;
	}
}
