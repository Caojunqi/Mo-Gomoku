package cn.caojunqi.game;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

import java.util.Random;

/**
 * 五子棋神经网络
 *
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class MctsBlock extends AbstractBlock {

    private NDManager netManager;
    private Random random;

    private Block commonLayers;
    private Block policyHeadConv;
    private Block policyHeadDense;
    private Block valueHeadConv;
    private Block valueHeadDense;

    public MctsBlock(NDManager netManager, Random random) {
        this.netManager = netManager;
        this.random = random;

        this.commonLayers = addChildBlock("common_layers", buildCommonLayers());
        this.policyHeadConv = addChildBlock("policy_head_conv", buildPolicyHeadConv());
        this.policyHeadDense = addChildBlock("policy_head_dense", buildPolicyHeadDense());
        this.valueHeadConv = addChildBlock("value_head_conv", buildValueHeadConv());
        this.valueHeadDense = addChildBlock("value_head_dense", buildValueHeadDense());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList x = this.commonLayers.forward(parameterStore, inputs, training);

        NDList policyConv = this.policyHeadConv.forward(parameterStore, x, training);
        NDList policyFlatten = new NDList(policyConv.get(0).reshape(policyConv.get(0).getShape().get(0), -1));
        NDArray policy = this.policyHeadDense.forward(parameterStore, policyFlatten, training).singletonOrThrow();
        policy = policy.logSoftmax(-1);

        NDList valueConv = this.valueHeadConv.forward(parameterStore, x, training);
        NDList valueFlatten = new NDList(valueConv.get(0).reshape(valueConv.get(0).getShape().get(0), -1));
        NDList vf = this.valueHeadDense.forward(parameterStore, valueFlatten, training);
        return new NDList(policy, vf.singletonOrThrow());
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{
                new Shape(1), new Shape(1), new Shape(Board.NUM_SQUARES)
        };
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {

        this.commonLayers.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.commonLayers.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.commonLayers.initialize(manager, dataType, inputShapes[0]);

        this.policyHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.policyHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.policyHeadConv.initialize(manager, dataType, new Shape(1, 128, Board.GRID_LENGTH, Board.GRID_LENGTH));

        this.policyHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.policyHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.policyHeadDense.initialize(manager, dataType, new Shape(4 * Board.NUM_SQUARES));

        this.valueHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.valueHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueHeadConv.initialize(manager, dataType, new Shape(1, 128, Board.GRID_LENGTH, Board.GRID_LENGTH));

        this.valueHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.valueHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueHeadDense.initialize(manager, dataType, new Shape(2 * Board.NUM_SQUARES));
    }

    public Block buildCommonLayers() {
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setFilters(32)
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder()
                        .optMomentum(0.9f)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder()
                        .optMomentum(0.9f)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(128)
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder()
                        .optMomentum(0.9f)
                        .build())
                .add(Activation::relu);
    }

    private Block buildPolicyHeadConv() {
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setFilters(4)
                        .setKernelShape(new Shape(1, 1))
                        .optStride(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder()
                        .optMomentum(0.9f)
                        .build())
                .add(Activation::relu);
    }

    private Block buildPolicyHeadDense() {
        return Linear.builder()
                .setUnits(Board.NUM_SQUARES)
                .build();
    }

    private Block buildValueHeadConv() {
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setFilters(2)
                        .setKernelShape(new Shape(1, 1))
                        .optStride(new Shape(1, 1))
                        .build())
                .add(BatchNorm.builder()
                        .optMomentum(0.9f)
                        .build())
                .add(Activation::relu);
    }

    private Block buildValueHeadDense() {
        return new SequentialBlock()
                .add(Linear.builder()
                        .setUnits(64)
                        .build())
                .add(Activation::relu)
                .add(Linear.builder()
                        .setUnits(1)
                        .build())
                .add(Activation::tanh);
    }
}
