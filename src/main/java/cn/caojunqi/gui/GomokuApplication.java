package cn.caojunqi.gui;

import ai.djl.ndarray.NDManager;
import cn.caojunqi.game.Board;
import cn.caojunqi.mcts.MctsTester;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

import java.util.Random;

/**
 * 五子棋应用
 *
 * @author Caojunqi
 * @date 2021-12-16 15:14
 */
public class GomokuApplication extends Application {

    private final static String GOMOKU_GUI_RESOURCE_DIR = "demo/gomoku/";

    @Override
    public void start(Stage primaryStage) throws Exception {
        Font.loadFont(getClass().getClassLoader().getResource
                (GOMOKU_GUI_RESOURCE_DIR + "FontAwesome.otf").toExternalForm(), 10);

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        Board gameEnv = new Board(mainManager.newSubManager(), random);
        MctsTester tester = new MctsTester(mainManager.newSubManager(), random, gameEnv, true);

        Pane gomokuPane = loadGomokuPane(mainManager, tester);

        primaryStage.setTitle("Gomoku");
        primaryStage.setScene(new Scene(gomokuPane, 800, 600));
        primaryStage.setMinWidth(800);
        primaryStage.setMinHeight(600);
        primaryStage.getIcons().add(new Image(getClass().getClassLoader()
                .getResource(GOMOKU_GUI_RESOURCE_DIR + "AppIcon.png").toExternalForm()));
        primaryStage.show();
    }

    private Pane loadGomokuPane(NDManager mainManager, MctsTester tester) {
        BorderPane gomokuPane = new BorderPane();
        GomokuBoardPane boardPane = new GomokuBoardPane(mainManager.newSubManager(), tester);
        gomokuPane.setCenter(boardPane);
        tester.getGameEnv().setBoardPane(boardPane);

        Button button = new Button("Start");
        Font font = Font.font("Consolas", FontWeight.BOLD, 13);
        button.setFont(font);
        button.setPrefWidth(100);
        button.setPrefHeight(100);
        button.setOnAction(value -> tester.start());
        gomokuPane.setLeft(button);
        return gomokuPane;
    }

}
