package mo.gomoku;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import mo.gomoku.game.Board;

/**
 * 五子棋神经网络
 *
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class MctsBlockTest extends AbstractBlock {
	private Block commonConv1;
	private Block commonConv2;
	private Block commonConv3;
	private Block policyConv;
	private Block policyDense;
	private Block valueConv;
	private Block valueDense1;
	private Block valueDense2;


//	private Block commonLayers;
//	private Block policyHeadConv;
//	private Block policyHeadDense;
//	private Block valueHeadConv;
//	private Block valueHeadDense;

	public MctsBlockTest() {
		this.commonConv1 = addChildBlock("common_conv_1", buildCommonConv1());
		this.commonConv2 = addChildBlock("common_conv_2", buildCommonConv2());
		this.commonConv3 = addChildBlock("common_conv_3", buildCommonConv3());
		this.policyConv = addChildBlock("policy_conv", buildPolicyConv());
		this.policyDense = addChildBlock("policy_dense", buildPolicyDense());
		this.valueConv = addChildBlock("value_conv", buildValueConv());
		this.valueDense1 = addChildBlock("value_dense_1", buildValueDense1());
		this.valueDense2 = addChildBlock("value_dense_2", buildValueDense2());


//		this.commonLayers = addChildBlock("common_layers", buildCommonLayers());
//		this.policyHeadConv = addChildBlock("policy_head_conv", buildPolicyHeadConv());
//		this.policyHeadDense = addChildBlock("policy_head_dense", buildPolicyHeadDense());
//		this.valueHeadConv = addChildBlock("value_head_conv", buildValueHeadConv());
//		this.valueHeadDense = addChildBlock("value_head_dense", buildValueHeadDense());
	}

	@Override
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
		NDList x = this.commonConv1.forward(parameterStore, inputs, training);
		x = Activation.relu(x);
		x = this.commonConv2.forward(parameterStore, x, training);
		x = Activation.relu(x);
		x = this.commonConv3.forward(parameterStore, x, training);
		x = Activation.relu(x);
		NDList xAct = this.policyConv.forward(parameterStore, x, training);
		xAct = Activation.relu(xAct);
		xAct = new NDList(xAct.get(0).reshape(xAct.get(0).getShape().get(0), -1));
		xAct = this.policyDense.forward(parameterStore, xAct, training);
		NDArray xActArr = xAct.singletonOrThrow().logSoftmax(-1);
		NDList xVal = this.valueConv.forward(parameterStore, x, training);
		xVal = Activation.relu(xVal);
		xVal = new NDList(xVal.get(0).reshape(xVal.get(0).getShape().get(0), -1));
		xVal = this.valueDense1.forward(parameterStore, xVal, training);
		xVal = Activation.relu(xVal);
		xVal = this.valueDense2.forward(parameterStore, xVal, training);
		xVal = Activation.tanh(xVal);
		return new NDList(xActArr, xVal.singletonOrThrow());


//
//
//		NDList x = this.commonLayers.forward(parameterStore, inputs, training);
//
//		NDList policyConv = this.policyHeadConv.forward(parameterStore, x, training);
//		NDList policyFlatten = new NDList(policyConv.get(0).reshape(policyConv.get(0).getShape().get(0), -1));
//		NDArray policy = this.policyHeadDense.forward(parameterStore, policyFlatten, training).singletonOrThrow();
//		policy = policy.logSoftmax(-1);
//
//		NDList valueConv = this.valueHeadConv.forward(parameterStore, x, training);
//		NDList valueFlatten = new NDList(valueConv.get(0).reshape(valueConv.get(0).getShape().get(0), -1));
//		NDList vf = this.valueHeadDense.forward(parameterStore, valueFlatten, training);
//		return new NDList(policy, vf.singletonOrThrow());
	}

	@Override
	public Shape[] getOutputShapes(Shape[] inputShapes) {
		return new Shape[]{
				new Shape(1), new Shape(1), new Shape(Board.NUM_SQUARES)
		};
	}

	@Override
	public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {

		this.commonConv1.setInitializer(new CommonConv1WeightInitializer(), Parameter.Type.WEIGHT);
		this.commonConv1.setInitializer(new CommonConv1BiasInitializer(), Parameter.Type.BIAS);
		this.commonConv1.initialize(manager, dataType, inputShapes[0]);
		this.commonConv2.setInitializer(new CommonConv2WeightInitializer(), Parameter.Type.WEIGHT);
		this.commonConv2.setInitializer(new CommonConv2BiasInitializer(), Parameter.Type.BIAS);
		this.commonConv2.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
		this.commonConv3.setInitializer(new CommonConv3WeightInitializer(), Parameter.Type.WEIGHT);
		this.commonConv3.setInitializer(new CommonConv3BiasInitializer(), Parameter.Type.BIAS);
		this.commonConv3.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
		this.policyConv.setInitializer(new PolicyConvWeightInitializer(), Parameter.Type.WEIGHT);
		this.policyConv.setInitializer(new PolicyConvBiasInitializer(), Parameter.Type.BIAS);
		this.policyConv.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
		this.policyDense.setInitializer(new PolicyDenseWeightInitializer(), Parameter.Type.WEIGHT);
		this.policyDense.setInitializer(new PolicyDenseBiasInitializer(), Parameter.Type.BIAS);
		this.policyDense.initialize(manager, dataType, new Shape(2 * Board.NUM_SQUARES));
		this.valueConv.setInitializer(new ValueConvWeightInitializer(), Parameter.Type.WEIGHT);
		this.valueConv.setInitializer(new ValueConvBiasInitializer(), Parameter.Type.BIAS);
		this.valueConv.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
		this.valueDense1.setInitializer(new ValueDense1WeightInitializer(), Parameter.Type.WEIGHT);
		this.valueDense1.setInitializer(new ValueDense1BiasInitializer(), Parameter.Type.BIAS);
		this.valueDense1.initialize(manager, dataType, new Shape(2 * Board.NUM_SQUARES));
		this.valueDense2.setInitializer(new ValueDense2WeightInitializer(), Parameter.Type.WEIGHT);
		this.valueDense2.setInitializer(new ValueDense2BiasInitializer(), Parameter.Type.BIAS);
		this.valueDense2.initialize(manager, dataType, new Shape(2));

//		this.commonLayers.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
//		this.commonLayers.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//		this.commonLayers.initialize(manager, dataType, inputShapes[0]);
//
//		this.policyHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
//		this.policyHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//		this.policyHeadConv.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
//
//		this.policyHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
//		this.policyHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//		this.policyHeadDense.initialize(manager, dataType, new Shape(2 * Board.NUM_SQUARES));
//
//		this.valueHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
//		this.valueHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//		this.valueHeadConv.initialize(manager, dataType, new Shape(1, 2, Board.GRID_LENGTH, Board.GRID_LENGTH));
//
//		this.valueHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
//		this.valueHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//		this.valueHeadDense.initialize(manager, dataType, new Shape(2 * Board.NUM_SQUARES));
	}

	private Block buildCommonConv1() {
		return Conv2d.builder()
				.setFilters(2)
				.setKernelShape(new Shape(3, 3))
				.optStride(new Shape(1, 1))
				.optPadding(new Shape(1, 1))
				.build();
	}

	private Block buildCommonConv2() {
		return Conv2d.builder()
				.setFilters(2)
				.setKernelShape(new Shape(3, 3))
				.optStride(new Shape(1, 1))
				.optPadding(new Shape(1, 1))
				.build();
	}

	private Block buildCommonConv3() {
		return Conv2d.builder()
				.setFilters(2)
				.setKernelShape(new Shape(3, 3))
				.optStride(new Shape(1, 1))
				.optPadding(new Shape(1, 1))
				.build();
	}

	private Block buildPolicyConv() {
		return Conv2d.builder()
				.setFilters(2)
				.setKernelShape(new Shape(1, 1))
				.optStride(new Shape(1, 1))
				.build();
	}

	private Block buildPolicyDense() {
		return Linear.builder()
				.setUnits(Board.NUM_SQUARES)
				.build();
	}

	private Block buildValueConv() {
		return Conv2d.builder()
				.setFilters(2)
				.setKernelShape(new Shape(1, 1))
				.optStride(new Shape(1, 1))
				.build();
	}

	private Block buildValueDense1() {
		return Linear.builder()
				.setUnits(2)
				.build();
	}

	private Block buildValueDense2() {
		return Linear.builder()
				.setUnits(1)
				.build();
	}

	public Block buildCommonLayers() {
		return new SequentialBlock()
				.add(Conv2d.builder()
						.setFilters(2)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(Activation::relu)
				.add(Conv2d.builder()
						.setFilters(2)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(Activation::relu)
				.add(Conv2d.builder()
						.setFilters(2)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(Activation::relu);
	}

	private Block buildPolicyHeadConv() {
		return new SequentialBlock()
				.add(Conv2d.builder()
						.setFilters(2)
						.setKernelShape(new Shape(1, 1))
						.optStride(new Shape(1, 1))
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
				.add(Activation::relu);
	}

	private Block buildValueHeadDense() {
		return new SequentialBlock()
				.add(Linear.builder()
						.setUnits(2)
						.build())
				.add(Activation::relu)
				.add(Linear.builder()
						.setUnits(1)
						.build())
				.add(Activation::tanh);
	}

	private static class CommonConv1WeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					0.0859f, -0.0736f, -0.0323f,
					0.0782f, -0.1569f, 0.1000f,
					-0.0343f, 0.0848f, 0.0232f,

					-0.0204f, 0.0462f, 0.0082f,
					0.0609f, -0.0650f, -0.0122f,
					-0.0150f, 0.0242f, -0.0007f,

					0.1457f, 0.0519f, -0.0621f,
					-0.1007f, -0.0279f, -0.0719f,
					-0.0534f, 0.0080f, 0.0994f,

					0.0906f, -0.1629f, 0.1033f,
					0.0466f, 0.1581f, 0.1100f,
					-0.1519f, -0.1585f, -0.0804f,


					0.1464f, -0.0278f, 0.0713f,
					-0.0775f, 0.1635f, -0.0705f,
					0.1250f, 0.0020f, -0.0878f,

					0.0857f, -0.0885f, 0.0490f,
					-0.0481f, -0.0183f, -0.1602f,
					-0.0795f, 0.0904f, -0.0405f,

					0.1660f, 0.1336f, -0.0078f,
					-0.1112f, 0.1015f, 0.0517f,
					-0.1077f, 0.1083f, 0.1012f,

					0.1478f, -0.0934f, -0.0274f,
					-0.0032f, 0.0243f, -0.1265f,
					-0.1183f, 0.0907f, -0.0391f
			};
			return manager.create(data).reshape(2, 4, 3, 3);
		}
	}

	private static class CommonConv1BiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.0814f, 0.0095f};
			return manager.create(data).reshape(2);
		}
	}

	private static class CommonConv2WeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					0.0774f, 0.0518f, 0.0857f,
					0.1168f, -0.2183f, 0.1186f,
					-0.1657f, -0.1778f, 0.0143f,

					-0.0402f, 0.1384f, -0.1365f,
					-0.2095f, 0.1715f, -0.0350f,
					0.1326f, 0.0758f, -0.1767f,


					0.0474f, 0.0566f, -0.1578f,
					-0.1118f, 0.0804f, 0.0422f,
					-0.1003f, -0.0714f, 0.2159f,

					-0.0436f, 0.1329f, 0.1021f,
					-0.1524f, -0.2004f, 0.2262f,
					0.0123f, 0.1616f, 0.0488f,
			};
			return manager.create(data).reshape(2, 2, 3, 3);
		}
	}

	private static class CommonConv2BiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.0758f, 0.1761f};
			return manager.create(data).reshape(2);
		}
	}

	private static class CommonConv3WeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					0.2235f, -0.1564f, 0.0295f,
					0.1759f, 0.1708f, 0.1464f,
					-0.1706f, -0.1697f, -0.1425f,

					0.0296f, 0.2349f, -0.1489f,
					0.1256f, -0.1305f, -0.2216f,
					-0.0501f, 0.1358f, 0.2188f,


					-0.1464f, 0.0512f, 0.2034f,
					0.1562f, 0.1469f, 0.1675f,
					0.1491f, 0.0609f, -0.1612f,

					-0.1979f, -0.1080f, -0.0274f,
					-0.1445f, 0.0862f, 0.0729f,
					-0.0534f, 0.0906f, 0.0762f
			};
			return manager.create(data).reshape(2, 2, 3, 3);
		}
	}

	private static class CommonConv3BiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.1439f, 0.1587f};
			return manager.create(data).reshape(2);
		}
	}

	private static class PolicyConvWeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					-0.2395f,
					0.6909f,
					-0.0817f,
					-0.0243f
			};
			return manager.create(data).reshape(2, 2, 1, 1);
		}
	}

	private static class PolicyConvBiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{-0.6674f, -0.4551f};
			return manager.create(data).reshape(2);
		}
	}

	private static class PolicyDenseWeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					-0.1377f, -0.1008f, 0.1676f, -0.0770f, -0.1761f, 0.0907f, 0.0755f, 0.1527f,
					-0.1220f, 0.0511f, -0.0858f, -0.0530f, -0.1878f, -0.1074f, -0.0722f, 0.1008f,
					0.0431f, 0.0582f,
					0.2353f, 0.2297f, 0.1608f, 0.0075f, -0.1631f, 0.1842f, -0.0589f, -0.0191f,
					-0.2031f, -0.0466f, -0.1520f, 0.2166f, -0.2038f, -0.1837f, -0.0080f, -0.1274f,
					0.0843f, -0.0907f,
					-0.1107f, 0.0133f, 0.1706f, -0.1658f, 0.1107f, 0.1514f, 0.2306f, -0.1650f,
					0.0571f, -0.1743f, 0.2012f, -0.0914f, 0.1420f, 0.0070f, -0.0184f, -0.0075f,
					0.0401f, 0.1111f,
					0.0378f, 0.0719f, -0.2120f, 0.1717f, 0.2055f, 0.1948f, 0.1742f, -0.1701f,
					-0.0874f, 0.2078f, -0.1795f, 0.2138f, -0.1854f, -0.1660f, 0.1152f, -0.1693f,
					-0.0540f, 0.1715f,
					0.1867f, 0.2229f, -0.0478f, -0.1832f, 0.2321f, -0.0502f, -0.0970f, 0.0575f,
					-0.1648f, 0.1549f, 0.1477f, -0.1870f, -0.1936f, -0.0206f, 0.0990f, -0.0068f,
					-0.1195f, 0.0054f,
					-0.2215f, -0.1666f, -0.1569f, 0.1941f, 0.2078f, -0.0801f, 0.0106f, 0.1051f,
					0.0282f, -0.1180f, 0.1359f, 0.1449f, -0.0137f, -0.0290f, 0.2142f, 0.2060f,
					-0.1336f, 0.2306f,
					0.0583f, -0.1566f, 0.1290f, -0.1760f, 0.2178f, -0.1515f, 0.0667f, 0.0718f,
					0.0561f, 0.1955f, -0.0979f, -0.0995f, -0.2043f, -0.0097f, -0.1117f, 0.0094f,
					-0.0483f, 0.0782f,
					0.2039f, 0.0696f, -0.0759f, -0.1157f, -0.2056f, 0.1983f, -0.0447f, 0.0476f,
					0.0087f, -0.1502f, 0.1327f, 0.1321f, 0.0119f, -0.1339f, -0.1002f, 0.0032f,
					-0.1347f, -0.1315f,
					-0.0352f, -0.0679f, 0.0579f, -0.0620f, -0.0312f, -0.0900f, -0.2154f, 0.2056f,
					0.0450f, 0.2108f, -0.0380f, -0.0592f, -0.2216f, -0.0895f, 0.1546f, -0.2309f,
					-0.0824f, 0.1436f
			};
			return manager.create(data).reshape(9, 18);
		}
	}

	private static class PolicyDenseBiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.1358f, 0.0870f, -0.0467f, -0.0178f, -0.0448f, -0.0869f, 0.0984f, -0.0803f, -0.1324f};
			return manager.create(data).reshape(9);
		}
	}

	private static class ValueConvWeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.1604f,
					-0.5111f,
					-0.1509f,
					0.3613f};
			return manager.create(data).reshape(2, 2, 1, 1);
		}
	}

	private static class ValueConvBiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.2643f, 0.6020f};
			return manager.create(data).reshape(2);
		}
	}

	private static class ValueDense1WeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{
					-0.1608f, 0.0020f, -0.0696f, -0.1318f, -0.1326f, 0.0594f, 0.0247f, -0.1203f,
					-0.1661f, -0.1953f, 0.2310f, -0.0402f, -0.0545f, -0.0867f, -0.1179f, -0.2156f,
					-0.1384f, 0.1441f,
					0.0517f, -0.0831f, -0.0891f, 0.1506f, 0.1699f, 0.2271f, 0.0688f, 0.1139f,
					-0.0189f, -0.1378f, -0.2309f, 0.1436f, -0.0343f, 0.0968f, 0.0116f, -0.2200f,
					-0.1156f, 0.0663f
			};
			return manager.create(data).reshape(2, 18);
		}
	}

	private static class ValueDense1BiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{-0.0477f, -0.2276f};
			return manager.create(data).reshape(2);
		}
	}

	private static class ValueDense2WeightInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{0.2480f, 0.0271f};
			return manager.create(data).reshape(1, 2);
		}
	}

	private static class ValueDense2BiasInitializer implements Initializer {
		@Override
		public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
			float[] data = new float[]{-0.2273f};
			return manager.create(data).reshape(1);
		}
	}
}
