package cn.caojunqi.game;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import cn.caojunqi.common.Tuple;
import cn.caojunqi.gui.GomokuBoardPane;
import cn.caojunqi.mcts.MctsSingleton;
import org.apache.commons.lang3.Validate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 五子棋棋盘
 *
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class Board {
	/**
	 * 棋盘长宽
	 */
	public static final int GRID_LENGTH = 6;
	/**
	 * 棋盘上格子数量
	 */
	public static final int NUM_SQUARES = GRID_LENGTH * GRID_LENGTH;
	/**
	 * X枚棋子连成一条直线后，游戏结束
	 */
	private static final int N_IN_ROW = 5;
	/**
	 * 玩家数量
	 */
	private static final int N_PLAYERS = 2;
	/**
	 * 落子信息
	 */
	private Map<Integer, Token> chessInfo;
	/**
	 * 合法的落子位置
	 */
	private List<Integer> availables;
	/**
	 * 当前已走步数
	 */
	private int turns;
	/**
	 * 最后一步所走的位置
	 */
	private int lastMove;
	/**
	 * 当前玩家索引
	 */
	private int curPlayerId;
	private GomokuBoardPane boardPane;

	public Board() {
		reset();
	}

	/**
	 * 游戏环境重置
	 */
	public void reset() {
		this.chessInfo = new HashMap<>(NUM_SQUARES);
		this.availables = buildAvailables();
		this.curPlayerId = 0;
		this.turns = 0;
		this.lastMove = -1;
		MctsSingleton.resetTempManager();
	}

	public void render() {
		if (boardPane != null) {
			boardPane.requestLayout();
		}
	}

	/**
	 * 构建最初始的合法落子位置集合
	 */
	private List<Integer> buildAvailables() {
		List<Integer> availables = new ArrayList<>(NUM_SQUARES);
		for (int i = 0; i < NUM_SQUARES; i++) {
			availables.add(i);
		}
		return availables;
	}

	/**
	 * @return 构建并返回当前棋盘状态
	 */
	public NDArray getCurState() {
		float[][] curPositions = new float[GRID_LENGTH][GRID_LENGTH];
		float[][] oppoPositions = new float[GRID_LENGTH][GRID_LENGTH];
		float[][] lastMove = new float[GRID_LENGTH][GRID_LENGTH];
		for (int i = 0; i < NUM_SQUARES; i++) {
			int h = i / GRID_LENGTH;
			int w = i % GRID_LENGTH;
			Token token = this.chessInfo.get(i);
			if (token != null) {
				if (this.curPlayerId == token.getPlayerId()) {
					curPositions[h][w] = 1;
				} else {
					oppoPositions[h][w] = 1;
				}
			}
		}

		if (this.lastMove != -1) {
			lastMove[this.lastMove / GRID_LENGTH][this.lastMove % GRID_LENGTH] = 1;
		}

		NDArray colourArr;
		if (this.chessInfo.size() % 2 == 0) {
			colourArr = MctsSingleton.TEMP_MANAGER.ones(new Shape(1, GRID_LENGTH, GRID_LENGTH), DataType.FLOAT32);
		} else {
			colourArr = MctsSingleton.TEMP_MANAGER.zeros(new Shape(1, GRID_LENGTH, GRID_LENGTH), DataType.FLOAT32);
		}

		NDArray curPositionArr = MctsSingleton.TEMP_MANAGER.create(curPositions).expandDims(0);
		NDArray oppoPositionArr = MctsSingleton.TEMP_MANAGER.create(oppoPositions).expandDims(0);
		NDArray lastMoveArr = MctsSingleton.TEMP_MANAGER.create(lastMove).expandDims(0);
		NDArray result = curPositionArr.
				concat(oppoPositionArr, 0).
				concat(lastMoveArr, 0).
				concat(colourArr, 0);
		result.attach(MctsSingleton.SAMPLE_MANAGER);
		return result;
	}

	public void doMove(int move) {
		Validate.isTrue(move < NUM_SQUARES);
		Validate.isTrue(!this.chessInfo.containsKey(move));
		this.chessInfo.put(move, Token.getPlayerToken(this.curPlayerId));
		this.availables.remove((Integer) move);
		this.turns++;
		this.curPlayerId = (this.curPlayerId + 1) % N_PLAYERS;
		this.lastMove = move;
		render();
	}

	public Shape getStateShape() {
		return new Shape(1, 4, GRID_LENGTH, GRID_LENGTH);
	}

	public Board deepCopy() {
		Board gameEnv = new Board();
		gameEnv.chessInfo = new HashMap<>(this.chessInfo);
		gameEnv.availables = new ArrayList<>(this.availables);
		gameEnv.turns = this.turns;
		gameEnv.lastMove = lastMove;
		gameEnv.curPlayerId = this.curPlayerId;
		gameEnv.boardPane = this.boardPane;
		return gameEnv;
	}

	/**
	 * 判断棋局是否结束
	 *
	 * @return first-棋局是否结束，true-结束，false-尚未结束；
	 * second-获胜者索引，-1表示没有获胜者，有可能是平局，有可能是因为棋局尚未结束。
	 */
	public Tuple<Boolean, Integer> checkGameOver() {
		if (this.turns < N_IN_ROW * 2 - 1) {
			// 行动步数太少，不可能有人获胜
			return new Tuple<>(false, -1);
		}

		for (Map.Entry<Integer, Token> entry : this.chessInfo.entrySet()) {
			int i = entry.getKey();
			Token token = entry.getValue();
			int h = i / GRID_LENGTH;
			int w = i % GRID_LENGTH;

			// 水平检测
			if (w < GRID_LENGTH - N_IN_ROW + 1) {
				boolean finish = true;
				for (int j = i; j < i + N_IN_ROW; j++) {
					if (!squareIsPlayer(j, token.playerId)) {
						finish = false;
						break;
					}
				}
				if (finish) {
					return new Tuple<>(true, token.playerId);
				}
			}

			// 垂直检测
			if (h < GRID_LENGTH - N_IN_ROW + 1) {
				boolean finish = true;
				for (int j = i; j < i + N_IN_ROW * GRID_LENGTH; j += GRID_LENGTH) {
					if (!squareIsPlayer(j, token.playerId)) {
						finish = false;
						break;
					}
				}
				if (finish) {
					return new Tuple<>(true, token.playerId);
				}
			}

			// 左上向右下检测
			if (w < GRID_LENGTH - N_IN_ROW + 1 && h < GRID_LENGTH - N_IN_ROW + 1) {
				boolean finish = true;
				for (int j = i; j < i + N_IN_ROW * (GRID_LENGTH + 1); j += (GRID_LENGTH + 1)) {
					if (!squareIsPlayer(j, token.playerId)) {
						finish = false;
						break;
					}
				}
				if (finish) {
					return new Tuple<>(true, token.playerId);
				}
			}

			// 右上向左下检测
			if (w > N_IN_ROW - 1 && h < GRID_LENGTH - N_IN_ROW + 1) {
				boolean finish = true;
				for (int j = i; j < i + N_IN_ROW * (GRID_LENGTH - 1); j += (GRID_LENGTH - 1)) {
					if (!squareIsPlayer(j, token.playerId)) {
						finish = false;
						break;
					}
				}
				if (finish) {
					return new Tuple<>(true, token.playerId);
				}
			}
		}

		if (this.turns == NUM_SQUARES) {
			return new Tuple<>(true, -1);
		}
		return new Tuple<>(false, -1);
	}

	/**
	 * 判断棋盘上指定位置是否为指定玩家落子
	 *
	 * @param square   棋盘位置索引
	 * @param playerId 目标玩家索引
	 * @return true-square上是指定玩家的落子；false-square上不是指定玩家的落子。
	 */
	private boolean squareIsPlayer(int square, int playerId) {
		Token token = this.chessInfo.get(square);
		return token != null && token.playerId == playerId;
	}

	public Map<Integer, Token> getChessInfo() {
		return chessInfo;
	}

	public List<Integer> getAvailables() {
		return availables;
	}

	public int getCurPlayerId() {
		return curPlayerId;
	}

	public void setBoardPane(GomokuBoardPane boardPane) {
		this.boardPane = boardPane;
	}

	/**
	 * 五子棋棋盘记号
	 */
	public enum Token {
		/**
		 * 无人落子
		 */
		NONE(".", -1),
		/**
		 * 玩家1落子
		 */
		X("X", 0),
		/**
		 * 玩家2落子
		 */
		O("O", 1),
		;

		public static final Token[] VALUES = Token.values();
		/**
		 * 字符串表示
		 */
		private String symbol;
		/**
		 * 对应的玩家索引
		 */
		private int playerId;

		Token(String symbol, int playerId) {
			this.symbol = symbol;
			this.playerId = playerId;
		}

		/**
		 * 获取指定玩家对应的棋盘记号
		 *
		 * @param playerId 玩家索引
		 * @return 该玩家使用的棋盘记号
		 */
		public static Token getPlayerToken(int playerId) {
			for (Token token : VALUES) {
				if (token.playerId == playerId) {
					return token;
				}
			}
			throw new IllegalArgumentException("不存在指定玩家索引对应的井字棋记号！！playerId:" + playerId);
		}

		public String getSymbol() {
			return symbol;
		}

		public int getPlayerId() {
			return playerId;
		}
	}

}
