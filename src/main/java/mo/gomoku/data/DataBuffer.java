package mo.gomoku.data;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import mo.gomoku.common.Triple;
import mo.gomoku.mcts.MctsSingleton;
import org.apache.commons.lang3.Validate;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 样本数据缓存
 *
 * @author Caojunqi
 * @date 2022-01-12 14:45
 */
public class DataBuffer {

	private int maxSize;
	private ArrayDeque<SampleData> datas;

	public DataBuffer(int maxSize) {
		this.maxSize = maxSize;
		this.datas = new ArrayDeque<>(maxSize);
	}

	public void cacheData(SampleData data) {
		Validate.isTrue(size() <= maxSize);
		if (size() == maxSize) {
			SampleData removeData = this.datas.removeFirst();
			removeData.close();
		}
		Validate.isTrue(size() < maxSize);
		this.datas.add(data);
	}

	/**
	 * 随机不重复地选出指定数量的样本
	 *
	 * @param count 需要的样本数量
	 * @return 随机采样结果 <状态压缩数组, 行为概率压缩数组, 胜者信息压缩数组>
	 */
	public Triple<NDArray, NDArray, NDArray> randomSample(int count) {
		Validate.isTrue(count > 0, "随机采样量必须大于0！！");
		Validate.isTrue(size() >= count, "样本数据容器中的数据量不足以支持随机采样！！totalSize[" + size() + "]，count[" + count + "]");
		List<SampleData> tmp = new ArrayList<>(datas);
		Collections.shuffle(tmp);
		List<SampleData> sampleResult = tmp.subList(0, count);

		NDManager manager = sampleResult.get(0).getState().getManager();

		NDList[] stateList = new NDList[count];
		NDList[] mctsProbList = new NDList[count];
		float[] winnerList = new float[count];
		for (int i = 0; i < count; i++) {
			SampleData sampleData = sampleResult.get(i);
			stateList[i] = new NDList(sampleData.getState());
			mctsProbList[i] = new NDList(sampleData.getMctsProbs());
			winnerList[i] = sampleData.getWinner();
		}

		return new Triple<>(Batchifier.STACK.batchify(stateList).singletonOrThrow(),
				Batchifier.STACK.batchify(mctsProbList).singletonOrThrow(),
				manager.create(winnerList));
	}

	public void clear() {
		System.out.println("开始清空样本====" + this.datas.size());
		for (SampleData data : this.datas) {
			data.close();
		}
		this.datas.clear();
		MctsSingleton.resetSampleManager();
	}

	/**
	 * 已收集的样本个数
	 */
	public int size() {
		return datas.size();
	}
}
