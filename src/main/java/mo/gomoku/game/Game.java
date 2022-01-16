package mo.gomoku.game;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import mo.gomoku.common.Tuple;
import mo.gomoku.data.PlayGameData;
import mo.gomoku.mcts.IAgent;
import mo.gomoku.mcts.MctsAlphaAgent;
import mo.gomoku.mcts.MctsPureAgent;
import mo.gomoku.mcts.MctsSingleton;

import java.util.ArrayList;
import java.util.List;

/**
 * 游戏类
 *
 * @author Caojunqi
 * @date 2022-01-11 21:19
 */
public class Game {

	private Board board;

	public Game(Board board) {
		this.board = board;
	}

	public int startEvaluatePlay(MctsAlphaAgent alphaAgent, MctsPureAgent pureAgent, boolean alphaFirst) {
		this.board.reset();
		IAgent[] agents = new IAgent[2];
		if (alphaFirst) {
			agents[0] = alphaAgent;
			agents[1] = pureAgent;
		} else {
			agents[0] = pureAgent;
			agents[1] = alphaAgent;
		}
		while (true) {
			IAgent agent = agents[this.board.getCurPlayerId()];
			int move = agent.getAction(this.board);
			this.board.doMove(move);
			Tuple<Boolean, Integer> gameResult = this.board.checkGameOver();
			if (gameResult.first) {
				return gameResult.second;
			}
		}
	}

	public PlayGameData startSelfPlay(MctsAlphaAgent agent) {
		this.board.reset();
		List<NDArray> states = new ArrayList<>();
		List<NDArray> mctsProbs = new ArrayList<>();
		List<Integer> currentPlayers = new ArrayList<>();
		while (true) {
			try (NDManager subManager = MctsSingleton.TEMP_MANAGER.newSubManager()) {
				Tuple<Integer, NDArray> action = agent.chooseAction(this.board, true);

				// store the data
				states.add(this.board.getCurState(subManager, MctsSingleton.SAMPLE_MANAGER));
				mctsProbs.add(action.second);
				currentPlayers.add(this.board.getCurPlayerId());

				// perform a move
				this.board.doMove(action.first);

				Tuple<Boolean, Integer> gameEndCheck = this.board.checkGameOver();
				if (gameEndCheck.first) {
					// winner from the perspective of the current player of each state
					float[] winners = new float[currentPlayers.size()];
					int winner = gameEndCheck.second;
					if (winner != -1) {
						for (int i = 0; i < currentPlayers.size(); i++) {
							if (currentPlayers.get(i) == winner) {
								winners[i] = 1.0f;
							} else {
								winners[i] = -1.0f;
							}
						}
					}
					agent.resetCore();
					return new PlayGameData(states, mctsProbs, winners);
				}
			}
		}
	}
}
