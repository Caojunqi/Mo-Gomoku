package cn.caojunqi.mcts;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import cn.caojunqi.common.Triple;
import cn.caojunqi.common.Tuple;
import cn.caojunqi.data.DataBuffer;
import cn.caojunqi.data.PlayGameData;
import cn.caojunqi.data.SampleData;
import cn.caojunqi.game.Board;
import cn.caojunqi.game.Game;
import cn.caojunqi.game.MctsBlock;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/**
 * 五子棋模型训练类
 *
 * @author Caojunqi
 * @date 2022-01-11 20:53
 */
public class MctsTrainer {
    private NDManager mainManager;
    private Random random;
    private Game game;
    private MctsLearningRateTracker tracker;
    private Trainer trainer;
    private MctsAlphaAgent agent;
    private DataBuffer dataBuffer;
    private float bestWinRatio;
    private int pureMctsPlayoutNum;

    public MctsTrainer() {
        Engine.getInstance().setRandomSeed(0);
        this.random = new Random(0);
        this.mainManager = NDManager.newBaseManager();
        Board board = new Board(this.mainManager.newSubManager(), this.random);
        this.game = new Game(board);
        this.tracker = new MctsLearningRateTracker();
        this.trainer = buildTrainer(board);
        this.agent = new MctsAlphaAgent(this.mainManager.newSubManager(), this.random, this.trainer);
        this.dataBuffer = new DataBuffer(MctsParameter.BUFFER_SIZE);
        this.pureMctsPlayoutNum = MctsParameter.PURE_MCTS_PLAYOUT_NUM;
    }

    public void run() {
        for (int i = 0; i < MctsParameter.GAME_BATCH_NUM; i++) {
            collectSelfplayData();
            if (this.dataBuffer.size() > MctsParameter.BATCH_SIZE) {
                Triple<NDArray, NDArray, NDArray> miniBatch = this.dataBuffer.randomSample(MctsParameter.BATCH_SIZE);
                trainBatch(miniBatch.first, miniBatch.second, miniBatch.third);
            }
            if ((i + 1) % MctsParameter.CHECK_FREQ == 0) {
                float winRatio = policyEvaluate();
                if (winRatio > this.bestWinRatio) {
                    this.bestWinRatio = winRatio;
                    // update the best policy
                    saveModel();
                    if (this.bestWinRatio == 1.0 && this.pureMctsPlayoutNum < MctsParameter.MAX_PURE_MCTS_PLAYOUT_NUM) {
                        this.pureMctsPlayoutNum += 1000;
                        this.bestWinRatio = 0.0f;
                    }
                }
            }
        }
    }

    private void trainBatch(NDArray stateBatch, NDArray mctsProbsBatch, NDArray winnerBatch) {
        NDList oldNetResult = this.trainer.forward(new NDList(stateBatch));
        NDArray oldLogActProbs = oldNetResult.get(0);
        NDArray oldActProbs = oldLogActProbs.duplicate().exp();
        NDArray oldValue = oldNetResult.get(1).duplicate();

        float kl = 0;
        for (int i = 0; i < MctsParameter.EPOCHS; i++) {
            Tuple<Float, Float> trainStepResult = trainStep(stateBatch, mctsProbsBatch, winnerBatch);
            float loss = trainStepResult.first;
            float entropy = trainStepResult.second;
            NDList newNetResult = this.trainer.forward(new NDList(stateBatch));
            NDArray newLogActProbs = newNetResult.get(0);
            NDArray newActProbs = newLogActProbs.duplicate().exp();
            NDArray newValue = newNetResult.get(1).duplicate();

            kl = oldActProbs.add(1e-10).log().sub(newActProbs.add(1e-10).log())
                    .mul(oldActProbs)
                    .sum(new int[]{1})
                    .mean().getFloat();
            if (kl > MctsParameter.KL_TARG * 4) {
                // early stopping if D_KL diverges badly
                break;
            }
        }

        // adaptively adjust the learning rate
        float lrMultiplier = this.tracker.getLrMultiplier();
        if (kl > MctsParameter.KL_TARG * 2 && lrMultiplier > 0.1) {
            lrMultiplier /= 1.5f;
        } else if (kl < MctsParameter.KL_TARG / 2 && lrMultiplier < 10) {
            lrMultiplier *= 1.5f;
        }
        this.tracker.setLrMultiplier(lrMultiplier);

        // TODO 此处还有一些统计信息
    }

    private void saveModel() {
        try {
            String fullDir = MctsParameter.MODEL_DIR +
                    MctsParameter.GAME_NAME +
                    MctsParameter.DIR_SEPARATOR;
            File modelFile = new File(fullDir);
            Path path = modelFile.toPath();
            this.trainer.getModel().save(path, MctsParameter.BEST_MODEL_PREFIX);
        } catch (IOException e) {
            throw new IllegalStateException("Best Model Save Error!!" + e);
        }
    }

    private Tuple<Float, Float> trainStep(NDArray stateBatch, NDArray mctsProbsBatch, NDArray winnerBatch) {
        NDList netResult = this.trainer.forward(new NDList(stateBatch));
        NDArray logActProbs = netResult.get(0);
        NDArray value = netResult.get(1);
        NDArray valueLoss = value.reshape(-1).sub(winnerBatch).square().mean();
        NDArray policyLoss = mctsProbsBatch.mul(logActProbs).sum(new int[]{1}).mean().neg();
        NDArray loss = valueLoss.add(policyLoss);
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            collector.backward(loss);
            trainer.step();
        }

        // calc policy entropy, for monitoring only
        NDArray entropy = logActProbs.exp().mul(logActProbs).sum(new int[]{1}).mean().neg();
        return new Tuple<>(loss.getFloat(), entropy.getFloat());
    }

    private Trainer buildTrainer(Board board) {
        Model model = buildBaseModel(this.mainManager.newSubManager(), random);
        TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Adam.builder().optLearningRateTracker(this.tracker).build());
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(board.getStateShape());
        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
        return trainer;
    }

    private Model buildBaseModel(NDManager manager, Random random) {
        Model policyModel = Model.newInstance(MctsParameter.GAME_NAME);
        MctsBlock policyNet = new MctsBlock(manager, random);
        policyModel.setBlock(policyNet);
        return policyModel;
    }

    /**
     * 收集“左右互搏”数据
     */
    private void collectSelfplayData() {
        for (int i = 0; i < MctsParameter.PLAY_BATCH_SIZE; i++) {
            PlayGameData playGameData = this.game.startSelfPlay(this.agent);
            augmentData(playGameData);
        }
    }

    /**
     * 对收集的对局数据进行增强（旋转、翻转），并填充进数据缓存器中
     *
     * @param playGameData 一局棋的对局数据
     */
    private void augmentData(PlayGameData playGameData) {
        int[] rotTimes = new int[]{1, 2, 3, 4};
        for (SampleData data : playGameData.getDatas()) {
            for (int i : rotTimes) {
                // 逆时针旋转90度
                NDArray augmentState = data.getState().rotate90(i, new int[]{1, 2});
                NDArray augmentMctsProbs = data.getMctsProbs().reshape(new Shape(Board.GRID_LENGTH, Board.GRID_LENGTH)).rotate90(i, new int[]{0, 1});
                this.dataBuffer.cacheData(new SampleData(augmentState, augmentMctsProbs.flatten(), data.getWinner()));

                //  水平翻转
                augmentState = augmentState.flip(2);
                augmentMctsProbs = augmentMctsProbs.flip(1);
                this.dataBuffer.cacheData(new SampleData(augmentState, augmentMctsProbs, data.getWinner()));
            }
        }
    }

    private float policyEvaluate() {
        MctsPureAgent pureAgent = new MctsPureAgent(this.random, this.pureMctsPlayoutNum);
        int winNum = 0;
        int tieNum = 0;
        for (int i = 0; i < MctsParameter.POLICY_EVALUATE_GAMES; i++) {
            boolean alphaFirst = i % 2 == 0;
            int winner = this.game.startEvaluatePlay(this.agent, pureAgent, alphaFirst);
            if (winner == -1) {
                tieNum++;
            } else {
                if (alphaFirst && winner == 0) {
                    winNum++;
                }
                if (!alphaFirst && winner == 1) {
                    winNum++;
                }
            }
        }
        return (1.0f * winNum + 0.5f * tieNum) / MctsParameter.POLICY_EVALUATE_GAMES;
    }
}