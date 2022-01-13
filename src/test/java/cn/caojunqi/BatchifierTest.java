package cn.caojunqi;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;

/**
 * 数组压缩测试类
 *
 * @author Caojunqi
 * @date 2022-01-12 17:20
 */
public class BatchifierTest {
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

        NDArray panelArr1 = manager.create(panel1);
        NDArray panelArr2 = manager.create(panel2);
        NDArray panelArr3 = manager.create(panel3);
        NDArray panelArr4 = manager.create(panel4);

        NDList[] allList = new NDList[4];
        allList[0] = new NDList(panelArr1);
        allList[1] = new NDList(panelArr2);
        allList[2] = new NDList(panelArr3);
        allList[3] = new NDList(panelArr4);

        NDArray arr = Batchifier.STACK.batchify(allList).singletonOrThrow();

        System.out.println("cc");

    }
}
