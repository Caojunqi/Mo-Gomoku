package cn.caojunqi.game;

import ai.djl.ndarray.NDArray;
import cn.caojunqi.common.Tuple;
import cn.caojunqi.data.PlayGameData;
import cn.caojunqi.mcts.IAgent;
import cn.caojunqi.mcts.MctsAlphaAgent;
import cn.caojunqi.mcts.MctsPureAgent;

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
			Tuple<Integer, NDArray> action = agent.chooseAction(this.board, true);

			// store the data
			states.add(this.board.getCurState());
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
