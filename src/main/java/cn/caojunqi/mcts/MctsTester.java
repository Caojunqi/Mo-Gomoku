package cn.caojunqi.mcts;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import cn.caojunqi.game.Board;
import cn.caojunqi.game.MctsBlock;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.Validate;

import java.io.File;
import java.util.*;

/**
 * 五子棋模型测试类
 *
 * @author Caojunqi
 * @date 2022-01-13 11:59
 */
public class MctsTester {

    /**
     * NDArray主管理器
     */
    private NDManager manager;
    /**
     * 随机数生成器
     */
    private Random random;
    /**
     * 游戏环境
     */
    private Board gameEnv;
    private int alphaPlayerId;
    /**
     * 对手智能体
     */
    private MctsAlphaAgent alphaAgent;
    /**
     * 对战是否开始
     */
    private boolean start;


    public MctsTester(NDManager manager, Random random, Board gameEnv, boolean alphaFirst) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.alphaPlayerId = alphaFirst ? 0 : 1;

        setupOpponents();
    }

    /**
     * 战斗开始
     */
    public void start() {
        if (this.start) {
            // 对战已经开始
            return;
        }
        gameEnv.reset();
        this.start = true;
        int curPlayerId = this.gameEnv.getCurPlayerId();
        while (curPlayerId == this.alphaPlayerId) {
            // 机器人行动
            int robotAction = this.alphaAgent.getAction(this.gameEnv);
            this.gameEnv.doMove(robotAction);
            curPlayerId = this.gameEnv.getCurPlayerId();
        }
    }

    /**
     * 玩家行动
     */
    public void playerAction(int action) {
        if (!this.start) {
            // 对战尚未开始
            return;
        }
        int curPlayerId = this.gameEnv.getCurPlayerId();
        if (curPlayerId == this.alphaPlayerId) {
            // 尚未到玩家行动
            return;
        }
        this.gameEnv.doMove(action);
        if (this.gameEnv.checkGameOver().first) {
            this.start = false;
            return;
        }
        curPlayerId = this.gameEnv.getCurPlayerId();
        while (curPlayerId == this.alphaPlayerId) {
            // 机器人行动
            int robotAction = this.alphaAgent.getAction(this.gameEnv);
            this.gameEnv.doMove(robotAction);
            curPlayerId = this.gameEnv.getCurPlayerId();
        }
        if (this.gameEnv.checkGameOver().first) {
            this.start = false;
        }
    }

    /**
     * 构建对手
     */
    private void setupOpponents() {
        Model opponentModel = loadOpponentModel();
        TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
        Trainer trainer = opponentModel.newTrainer(config);
        trainer.initialize(this.gameEnv.getStateShape());
        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
        this.alphaAgent = new MctsAlphaAgent(this.manager.newSubManager(), this.random, trainer);
    }

    /**
     * 加载训练好的模型
     */
    private Model loadOpponentModel() {
        File modelDir = new File(MctsParameter.MODEL_DIR + MctsParameter.GAME_NAME + MctsParameter.DIR_SEPARATOR);
        Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
        Validate.isTrue(!modelFiles.isEmpty(), "没有训练好的模型，不能进行性能测试！！");
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

    private Model buildBaseModel() {
        Model policyModel = Model.newInstance(MctsParameter.GAME_NAME);
        MctsBlock policyNet = new MctsBlock(this.manager.newSubManager(), this.random);
        policyModel.setBlock(policyNet);
        return policyModel;
    }

    public Board getGameEnv() {
        return gameEnv;
    }
}
