package cn.caojunqi.mcts;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.Trainer;
import cn.caojunqi.common.Tuple;
import cn.caojunqi.game.Board;
import cn.caojunqi.util.GomokuUtils;
import jsat.distributions.multivariate.Dirichlet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * 蒙特卡洛树搜索，利用神经网络进行状态价值评估
 *
 * @author Caojunqi
 * @date 2021-12-23 11:14
 */
public class MctsAlphaAgent implements IAgent {

	private Trainer trainer;
	private MctsAlphaCore core;

	public MctsAlphaAgent(Trainer trainer) {
		this.trainer = trainer;

		Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn = board -> {
			NDArray state = board.getCurState();
			NDList netResult = this.trainer.forward(new NDList(state));
			NDArray logActProbs = netResult.get(0);
			NDArray value = netResult.get(1);
			float[] allActProbs = logActProbs.exp().toFloatArray();
			List<Integer> availables = board.getAvailables();
			Map<Integer, Float> actProbs = new HashMap<>(availables.size());
			for (int available : availables) {
				actProbs.put(available, allActProbs[available]);
			}
			return new Tuple<>(actProbs, value.getFloat());
		};
		this.core = new MctsAlphaCore(policyValueFn);
	}

	@Override
	public int getAction(Board board) {
		Tuple<Integer, NDArray> chooseResult = chooseAction(board, false);
		return chooseResult.first;
	}

	public Tuple<Integer, NDArray> chooseAction(Board board, boolean training) {
		Tuple<NDArray, NDArray> actProbs = this.core.getMoveProbs(board);
		NDArray moveProbs = MctsSingleton.TEMP_MANAGER.zeros(new Shape(Board.NUM_SQUARES));
		int availableActionSize = board.getAvailables().size();
		for (int i = 0; i < availableActionSize; i++) {
			moveProbs.set(new NDIndex(actProbs.first.getInt(i)), actProbs.second.getFloat(i));
		}
		int action;
		if (training) {
			NDArray ndArr = MctsSingleton.TEMP_MANAGER.ones(new Shape(availableActionSize), DataType.FLOAT64);
			ndArr.muli(0.3);
			DenseVector denseVector = new DenseVector(ndArr.toDoubleArray());
			Dirichlet dirichlet = new Dirichlet(denseVector);
			List<Vec> dirichletSample = dirichlet.sample(1, MctsSingleton.RANDOM);
			double[] dirichletResult = dirichletSample.get(0).arrayCopy();
			NDArray dirichletRandomArr = MctsSingleton.TEMP_MANAGER.create(dirichletResult).toType(DataType.FLOAT32, false);
			NDArray finalActProbs = actProbs.second.mul(0.75).add(dirichletRandomArr.mul(0.25));
			int actionIndex = GomokuUtils.sampleMultinomial(finalActProbs);
			action = actProbs.first.get(actionIndex).getInt();
			this.core.updateWithMove(action);
		} else {
			int actionIndex = GomokuUtils.sampleMultinomial(actProbs.second);
			action = actProbs.first.get(actionIndex).getInt();
			this.core.updateWithMove(-1);
		}
		moveProbs.attach(MctsSingleton.SAMPLE_MANAGER);
		return new Tuple<>(action, moveProbs);
	}

	/**
	 * 重置搜索树核心
	 */
	public void resetCore() {
		this.core.updateWithMove(-1);
	}

	public void close() {
		this.trainer.close();
	}
}
