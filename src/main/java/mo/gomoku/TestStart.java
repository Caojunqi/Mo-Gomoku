package mo.gomoku;

import ai.djl.engine.Engine;
import javafx.application.Application;
import mo.gomoku.gui.GomokuApplication;

/**
 * 模型性能测试类
 *
 * @author Caojunqi
 * @date 2022-01-13 11:55
 */
public class TestStart {

	public static void main(String[] args) {
		Engine.getInstance().setRandomSeed(0);
		Application.launch(GomokuApplication.class, args);
	}

}
