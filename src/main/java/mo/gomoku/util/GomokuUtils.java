package mo.gomoku.util;

import ai.djl.ndarray.NDArray;
import mo.gomoku.mcts.MctsSingleton;
import org.apache.commons.lang3.Validate;

/**
 * 五子棋工具类
 *
 * @author Caojunqi
 * @date 2022-01-11 20:51
 */
public final class GomokuUtils {

	/**
	 * 从离散型动作分布中随机抽取一个动作
	 *
	 * @param distribution 动作分布，其数据长度就是可选动作的数量，数据值表示该index的动作被选中的概率
	 * @return 选中的动作
	 */
	public static int sampleMultinomial(NDArray distribution) {
		Validate.isTrue(distribution.getShape().dimension() == 1);
		int value = 0;
		long size = distribution.size();
		float rnd = MctsSingleton.RANDOM.nextFloat();
		for (int i = 0; i < size; i++) {
			float cut = distribution.getFloat(value);
			if (rnd > cut) {
				value++;
			} else {
				return value;
			}
			rnd -= cut;
		}

		throw new IllegalArgumentException("Invalid multinomial distribution");
	}
}
