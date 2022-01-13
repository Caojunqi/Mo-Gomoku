package cn.caojunqi.data;

import ai.djl.ndarray.NDArray;

/**
 * 单个样本数据
 *
 * @author Caojunqi
 * @date 2022-01-12 14:41
 */
public class SampleData {
	/**
	 * 单个棋盘状态
	 */
	private NDArray state;
	/**
	 * 当前棋盘状态下的动作选择概率
	 */
	private NDArray mctsProbs;
	/**
	 * 1：当前状态最终获胜
	 * -1：当前状态最终失败
	 */
	private float winner;

	public SampleData(NDArray state, NDArray mctsProbs, float winner) {
		this.state = state;
		this.mctsProbs = mctsProbs;
		this.winner = winner;
	}

	public NDArray getState() {
		return state;
	}

	public NDArray getMctsProbs() {
		return mctsProbs;
	}

	public float getWinner() {
		return winner;
	}
}
