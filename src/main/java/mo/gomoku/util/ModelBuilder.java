package mo.gomoku.util;

import ai.djl.Model;
import mo.gomoku.game.MctsBlock;
import mo.gomoku.mcts.MctsParameter;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

/**
 * 神经网络模型构建类
 *
 * @author Caojunqi
 * @date 2022-01-14 12:18
 */
public final class ModelBuilder {

	/**
	 * 从训练好的模型中选出最好的模型
	 */
	public static Model buildBestModel() {
		File modelDir = new File(MctsParameter.MODEL_DIR + MctsParameter.GAME_NAME + MctsParameter.DIR_SEPARATOR);
		Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
		if (modelFiles.isEmpty()) {
			return null;
		}
		List<File> sortedFiles = new ArrayList<>(modelFiles);
		sortedFiles.sort(Comparator.comparing(File::getName));
		try {
			File bestModelFileDir = new File(MctsParameter.MODEL_DIR + MctsParameter.GAME_NAME + MctsParameter.DIR_SEPARATOR);
			Model bestModel = buildBaseModel();
			bestModel.load(bestModelFileDir.toPath(), MctsParameter.BEST_MODEL_PREFIX, null);
			return bestModel;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * 构建一个参数随机的模型
	 */
	public static Model buildBaseModel() {
		Model policyModel = Model.newInstance(MctsParameter.GAME_NAME);
		MctsBlock policyNet = new MctsBlock();
		policyModel.setBlock(policyNet);
		return policyModel;
	}
}
