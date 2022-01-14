package mo.gomoku.data;

import ai.djl.ndarray.NDArray;
import org.apache.commons.lang3.Validate;

import java.util.ArrayList;
import java.util.List;

/**
 * 一场对局的数据
 *
 * @author Caojunqi
 * @date 2022-01-11 21:25
 */
public class PlayGameData {

	private List<SampleData> datas;

	public PlayGameData(List<NDArray> states, List<NDArray> mctsProbs, float[] winners) {
		Validate.isTrue(states.size() == mctsProbs.size() && states.size() == winners.length, "一局棋数据有误！！");
		int size = states.size();
		this.datas = new ArrayList<>(size);
		for (int i = 0; i < size; i++) {
			this.datas.add(new SampleData(states.get(i), mctsProbs.get(1), winners[i]));
		}
	}

	public List<SampleData> getDatas() {
		return datas;
	}
}
