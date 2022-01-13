package cn.caojunqi.mcts;

import cn.caojunqi.game.Board;

/**
 * 游戏主体
 *
 * @author Caojunqi
 * @date 2022-01-13 11:12
 */
public interface IAgent {

	int getAction(Board board);
}
