package mo.gomoku.mcts;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import mo.gomoku.common.Tuple;
import mo.gomoku.game.Board;
import org.apache.commons.lang3.Validate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * 蒙特卡洛树搜索逻辑核心
 *
 * @author Caojunqi
 * @date 2021-12-23 11:56
 */
public class MctsAlphaCore {

	private TreeNode root;
	private Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn;

	public MctsAlphaCore(Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn) {
		this.root = new TreeNode(null, 1.0f);
		this.policyValueFn = policyValueFn;
	}

	/**
	 * Run all playouts sequentially and return the available actions and
	 * their corresponding probabilities.
	 *
	 * @param gameEnv the current game state
	 * @return
	 */
	public Tuple<NDArray, NDArray> getMoveProbs(Board gameEnv) {
		for (int i = 0; i < MctsParameter.N_PLAYOUT; i++) {
			Board gameEnvCopy = gameEnv.deepCopy();
			playout(gameEnvCopy);
		}

		// calc the move probabilities based on visit counts at the root node
		List<Map.Entry<Integer, TreeNode>> entries = new ArrayList<>(this.root.getChildren().entrySet());
		int[] acts = new int[entries.size()];
		int[] visits = new int[entries.size()];
		for (int i = 0; i < entries.size(); i++) {
			Map.Entry<Integer, TreeNode> entry = entries.get(i);
			acts[i] = entry.getKey();
			visits[i] = entry.getValue().getVisits();
		}
		NDArray actsArr = MctsSingleton.TEMP_MANAGER.create(acts);
		NDArray visitsArr = MctsSingleton.TEMP_MANAGER.create(visits);
		visitsArr = visitsArr.add(1e-10).log().mul(1 / MctsParameter.MCTS_TEMP).softmax(-1).toType(DataType.FLOAT32, false);
		return new Tuple<>(actsArr, visitsArr);
	}

	/**
	 * Step forward in the tree, keeping everything we already know about the subtree.
	 *
	 * @param lastMove
	 */
	public void updateWithMove(int lastMove) {
		TreeNode child = this.root.getChildren().get(lastMove);
		if (child != null) {
			this.root = child;
			this.root.setParent(null);
		} else {
			Validate.isTrue(lastMove == -1);
			this.root = new TreeNode(null, 1);
		}
	}

	/**
	 * Run a single playout from the root to the leaf, getting a value at
	 * the leaf and propagating it back through its parents.
	 * State is modified in-place, so a copy must be provided.
	 *
	 * @param board
	 */
	private void playout(Board board) {
		TreeNode node = this.root;
		while (!node.isLeaf()) {
			// Greedily select next move.
			Tuple<Integer, TreeNode> selectResult = node.select();
			board.doMove(selectResult.first);
			node = selectResult.second;
		}

		Tuple<Map<Integer, Float>, Float> policy = this.policyValueFn.apply(board);
		Tuple<Boolean, Integer> gameResult = board.checkGameOver();
		float leafValue = policy.second;

		if (!gameResult.first) {
			node.expand(policy.first);
		} else {
			if (gameResult.second == -1) {
				leafValue = 0.0f;
			} else {
				leafValue = gameResult.second == board.getCurPlayerId() ? 1.0f : -1.0f;
			}
		}
		node.updateRecursive(-leafValue);
	}


}
