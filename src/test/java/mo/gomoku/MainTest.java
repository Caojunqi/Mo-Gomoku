package mo.gomoku;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * 主测试类
 *
 * @author Caojunqi
 * @date 2022-01-11 20:33
 */
public class MainTest {

	public static void main(String[] args) {
		NDManager manager = NDManager.newBaseManager();

		int[][] data = new int[][]{
				{1, 2},
				{3, 4}
		};

		NDArray array = manager.create(data);
		NDArray newArr = array.reshape(-1);
		System.out.println("cc");
	}
}
