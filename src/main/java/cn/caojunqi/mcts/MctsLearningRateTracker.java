package cn.caojunqi.mcts;

import ai.djl.training.tracker.Tracker;

/**
 * 学习率更新器
 *
 * @author Caojunqi
 * @date 2022-01-12 20:50
 */
public class MctsLearningRateTracker implements Tracker {

    private float lrMultiplier;

    public MctsLearningRateTracker() {
        this.lrMultiplier = MctsParameter.LR_MULTIPLIER;
    }

    @Override
    public float getNewValue(int numUpdate) {
        return MctsParameter.LEARN_RATE * lrMultiplier;
    }

    public float getLrMultiplier() {
        return lrMultiplier;
    }

    public void setLrMultiplier(float lrMultiplier) {
        this.lrMultiplier = lrMultiplier;
    }
}
