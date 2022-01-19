package mo.gomoku.mcts;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

import java.util.Random;

/**
 * 全局唯一对象管理类
 *
 * @author Caojunqi
 * @date 2022-01-13 21:48
 */
public class MctsSingleton {
	/**
	 * 一般运算过程中生成的数组都放在CPU设备中
	 */
	public static Device CPU_DEVICE = Device.cpu();
	/**
	 * 全局共用的随机数生成器
	 */
	public static Random RANDOM = new Random(0);
	/**
	 * 样本数据资源管理器，所有样本数据中的数组资源均由此管理。
	 */
	public static NDManager SAMPLE_MANAGER = NDManager.newBaseManager(CPU_DEVICE);
	/**
	 * 临时数据资源管理类，所有临时数组均由此管理，定时清空。
	 */
	public static NDManager TEMP_MANAGER = NDManager.newBaseManager(CPU_DEVICE);
	/**
	 * 神经网络资源管理类
	 */
	public static NDManager NET_MANAGER = NDManager.newBaseManager(QUICK_DEVICE);
	/**
	 * 是否启用GPU加速
	 */
	private static boolean GPU_QUICK = true;
	/**
	 * 模型训练放在加速设备中
	 */
	public static Device QUICK_DEVICE = GPU_QUICK ? Engine.getInstance().defaultDevice() : CPU_DEVICE;

	public static void resetSampleManager() {
		SAMPLE_MANAGER.close();
		SAMPLE_MANAGER = NDManager.newBaseManager(CPU_DEVICE);
	}

	public static void resetTempManager() {
		TEMP_MANAGER.close();
		TEMP_MANAGER = NDManager.newBaseManager(CPU_DEVICE);
	}
}
