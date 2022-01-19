package mo.gomoku.mcts;

/**
 * 蒙特卡洛树搜索常量类
 *
 * @author Caojunqi
 * @date 2022-01-11 15:56
 */
public final class MctsParameter {
	public static final String GAME_NAME = "gomoku";
	/**
	 * num of simulations for each move
	 */
	public static final int N_PLAYOUT = 400;
	/**
	 * a number in (0, inf) controlling the relative impact of value {@link TreeNode} Q, and {@link TreeNode} prior probability on this node's score.
	 */
	public static final int C_PUCT = 5;

	public static final float TRAIN_MCTS_TEMP = 1.0f;

	public static final float EVALUATE_MCTS_TEMP = 1e-3f;

	public static final float LEARN_RATE = 2e-3f;

	public static final float LR_MULTIPLIER = 1.0f;
	/**
	 * coef of l2 penalty
	 */
	public static final float L2_CONST = 1e-4f;

	public static final int GAME_BATCH_NUM = 1500000;

	public static final int PLAY_BATCH_SIZE = 1;

	public static final int BUFFER_SIZE = 10000;

	public static final int BATCH_SIZE = 512;
	/**
	 * num of train steps for each update
	 */
	public static final int EPOCHS = 5;

	public static final float KL_TARG = 0.02f;

	public static final int CHECK_FREQ = 50;
	/**
	 * num of simulations used for the pure mcts, which is used as
	 * the opponent to evaluate the trained policy
	 */
	public static final int PURE_MCTS_PLAYOUT_NUM = 1000;

	public static final int MAX_PURE_MCTS_PLAYOUT_NUM = 5000;

	public static final String MODEL_DIR = "src/main/resources/model/";

	public static final String BEST_MODEL_PREFIX = "best";
	/**
	 * 文件夹分隔符
	 */
	public final static String DIR_SEPARATOR = "/";

	public final static int EVALUATE_ROLLOUT_LIMIT = 1000;

	public final static int POLICY_EVALUATE_GAMES = 10;
}
