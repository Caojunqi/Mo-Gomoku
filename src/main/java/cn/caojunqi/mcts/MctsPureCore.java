package cn.caojunqi.mcts;

import cn.caojunqi.common.Tuple;
import cn.caojunqi.game.Board;
import org.apache.commons.lang3.Validate;

import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/**
 * 蒙特卡洛树搜索逻辑核心
 *
 * @author Caojunqi
 * @date 2021-12-23 11:56
 */
public class MctsPureCore {

    private Random random;
    private TreeNode root;
    private Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn;
    private int playout;

    public MctsPureCore(Random random, Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn, int playout) {
        this.random = random;
        this.root = new TreeNode(null, 1.0f);
        this.policyValueFn = policyValueFn;
        this.playout = playout;
    }

    /**
     * Run all playouts sequentially and returns the most visited action.
     *
     * @param board the current game state
     * @return the selected action
     */
    public int getMove(Board board) {
        for (int i = 0; i < this.playout; i++) {
            Board boardCopy = board.deepCopy();
            playout(boardCopy);
        }

        int bestMove = -1;
        int mostVisit = -1;
        for (Map.Entry<Integer, TreeNode> child : this.root.getChildren().entrySet()) {
            int childVisit = child.getValue().getVisits();
            if (childVisit > mostVisit) {
                mostVisit = childVisit;
                bestMove = child.getKey();
            }
        }
        Validate.isTrue(bestMove != -1);
        return bestMove;
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
        Map<Integer, Float> actionProbs = this.policyValueFn.apply(board).first;
        // Check for end of game
        Tuple<Boolean, Integer> gameResult = board.checkGameOver();
        if (!gameResult.first) {
            node.expand(actionProbs);
        }
        // Evaluate the leaf node by random rollout
        float leafValue = evaluateRollout(board);
        node.updateRecursive(-leafValue);
    }

    /**
     * Use the rollout policy to play until the end of the game,
     * returning +1 if the current player wins, -1 if the opponent wins,
     * and 0 if it is a ite.
     *
     * @param board
     * @return
     */
    private float evaluateRollout(Board board) {
        int curPlayerId = board.getCurPlayerId();
        Tuple<Boolean, Integer> gameResult = null;
        for (int i = 0; i < MctsParameter.EVALUATE_ROLLOUT_LIMIT; i++) {
            gameResult = board.checkGameOver();
            if (gameResult.first) {
                break;
            }
            int maxAction = rolloutPolicyFn(board);
            board.doMove(maxAction);
        }
        Validate.notNull(gameResult);
        if (!gameResult.first) {
            System.out.println("WARNING: rollout reached move limit");
        }
        int winner = gameResult.second;
        if (winner == -1) {
            return 0;
        }
        return winner == curPlayerId ? 1 : -1;
    }

    /**
     * a coarse, fast version of policy fn used in the rollout phase.
     *
     * @param board
     * @return
     */
    private int rolloutPolicyFn(Board board) {
        // rollout randomly
        float maxProb = -1;
        int maxAction = -1;
        for (int action : board.getAvailables()) {
            float randomProb = this.random.nextFloat();
            if (randomProb > maxProb) {
                maxProb = randomProb;
                maxAction = action;
            }
        }
        Validate.isTrue(maxAction != -1);
        return maxAction;
    }


}
