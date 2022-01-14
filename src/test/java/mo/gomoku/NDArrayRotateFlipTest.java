package mo.gomoku;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * 数组旋转、翻转测试类
 *
 * @author Caojunqi
 * @date 2022-01-12 16:43
 */
public class NDArrayRotateFlipTest {

	public static void main(String[] args) {
		NDManager manager = NDManager.newBaseManager();

		int[][] panel1 = new int[][]{
				{1, 2},
				{3, 4}
		};
		int[][] panel2 = new int[][]{
				{11, 12},
				{13, 14}
		};
		int[][] panel3 = new int[][]{
				{21, 22},
				{23, 24}
		};
		int[][] panel4 = new int[][]{
				{31, 32},
				{33, 34}
		};

		NDArray panelArr1 = manager.create(panel1).expandDims(0);
		NDArray panelArr2 = manager.create(panel2).expandDims(0);
		NDArray panelArr3 = manager.create(panel3).expandDims(0);
		NDArray panelArr4 = manager.create(panel4).expandDims(0);

		NDArray oldArr = panelArr1
				.concat(panelArr2)
				.concat(panelArr3)
				.concat(panelArr4);
		System.out.println(oldArr);

//        NDArray newArr = oldArr.rotate90(1, new int[]{1, 2});
		NDArray newArr = oldArr.flip(2);
		System.out.println(newArr.toDebugString());

		System.out.println("cc");

	}

}
