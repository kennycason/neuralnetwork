package nn;

/**
 * 多数中間層のニューラルネットのクラス（誤差逆伝播法）: multi-center layer neural network (back-error
 * propagation algorithm)
 * 
 * @author kenneth cason
 */
public class NeuralNetwork {
	
	private Layer inputLayer; // 入力層: input layer
	
	private Layer[] centerLayers; // 中間層: middle layers
	
	private Layer outputLayer; // 出力層: output layer

	private NeuralNetworkConfig config;

	public NeuralNetwork(NeuralNetworkConfig config) {
		this.config = config;
		init();
	}

	/**
	 * ニューラルネットを初期化する: initialize the neural network
	 * 
	 * @param config
	 *            .numInputNodes 入力層のノード数: number of nodes in the input layer
	 * @param numCL
	 *            中間層数: number of center layers
	 * @param config
	 *            .numCenterNodes 中間層のノード数: number of nodes in the center layer
	 * @param numOuputNodes
	 *            出力層のノード数: number of nodes in the output layer
	 * @param config
	 *            .bias
	 */
	private void init() {
		inputLayer = new Layer();
		centerLayers = new Layer[config.numCenterLayers];
		for (int i = 0; i < config.numCenterLayers; i++) {
			centerLayers[i] = new Layer();
		}
		outputLayer = new Layer();

		// 出力層: output layer
		outputLayer.init(config.numOutputNodes,
				centerLayers[config.numCenterLayers - 1], null, config.bias);
		// System.out.println("OUTPUT INITED");
		// 中間層: middle layer
		for (int i = config.numCenterLayers - 1; i >= 0; i--) {
			if (i == 0) {
				if (config.numCenterLayers == 1) {// もし中間層の最後の層だったら、出力層と繋がる; if
													// it is the last of the
													// center layers, connect to
													// the output layer
					centerLayers[i].init(config.numCenterNodes, inputLayer,
							outputLayer, config.bias); // 中間層数が一だから、親層＝入力層、子層＝出力層:
														// because there is only
														// one center layer,
														// parent layer = input
														// layer, child layer =
														// outputlayer
				} else {
					centerLayers[i].init(config.numCenterNodes, inputLayer,
							centerLayers[i + 1], config.bias);
				}
			} else { // 前層は入力層ではない: previous layer does not have an input layer
				if (i == config.numCenterLayers - 1) {// もし中間層の最後の層だったら、出力層と繋がる;
														// if it is the last of
														// the center layers,
														// connect to the output
														// layer
					centerLayers[i].init(config.numCenterNodes,
							centerLayers[i - 1], outputLayer, config.bias);
				} else {
					centerLayers[i].init(config.numCenterNodes,
							centerLayers[i - 1], centerLayers[i + 1],
							config.bias);
				}
			}
		}
		// 入力層: input layer
		inputLayer.init(config.numInputNodes, null, centerLayers[0],
				config.bias);
		setLearningRate(config.learningRate);
	}

	/**
	 * 入力層から出力層まで前向きを伝播する: propagate from the input layer to the output layer
	 */
	public void feedForward() {
		inputLayer.calculateNeuronValues();
		for (Layer l : centerLayers) {
			l.calculateNeuronValues();
		}
		outputLayer.calculateNeuronValues();
	}

	/**
	 * 出力層から入力層まで逆向きに伝播する: back-propagate from the output layer to the input
	 * layer
	 */
	public void backPropagate() {
		outputLayer.calculateErrors();
		for (int i = centerLayers.length - 1; i >= 0; i--) {
			centerLayers[i].calculateErrors();
			centerLayers[i].adjustWeights();
		}
		inputLayer.adjustWeights();
	}

	/**
	 * 出力と教師信号の平均２乗誤差を計算する: calculate the average squared error between the
	 * output layer and teacher signal
	 * 
	 * @return 平均２乗誤差
	 */
	public double calculateError() {
		double error = 0;
		for (int i = 0; i < outputLayer.getNumNeurons(); i++) {
			error += Math.pow(
					outputLayer.getNeuron(i).getValue()
							- outputLayer.getTeacherSignal(i), 2);
		}
		error /= outputLayer.getNumNeurons();
		return error;
	}

	/**
	 * 各層の各ニューロンの値をゼロにする: clear each layer's neuron values
	 */
	public void clearAllValues() {
		outputLayer.clearAllValues();
		for (int i = centerLayers.length - 1; i >= 0; i--) {
			centerLayers[i].clearAllValues();
		}
		inputLayer.clearAllValues();
	}

	/**
	 * 　入力層への一つの入力を設定する: set one value in the input layer
	 * 
	 * @param i
	 *            ノード番号: node number
	 * @param value
	 *            値: value
	 */
	public void setInput(int i, double value) {
		if (i >= 0 && i < inputLayer.getNumNeurons()) {
			inputLayer.getNeuron(i).setValue(value);
		}
	}

	/**
	 * 　入力層への各入力を設定する: set all values in the input layer
	 * 
	 * @param values
	 *            値: value
	 */
	public void setInputs(double[] values) {
		if (inputLayer.getNumNeurons() == values.length) {
			for (int i = 0; i < inputLayer.getNumNeurons(); i++) {
				inputLayer.getNeuron(i).setValue(values[i]);
			}
		} else {
			System.out.println("The Input dimensions do not match precisely.");
		}
	}

	/**
	 * 　入力層への各入力を設定する: set all values in the input layer
	 * 
	 * @param values
	 *            値: value
	 */
	public void setInputs(double[][] values) {
		for (int y = 0; y < config.inputHeight
				&& y < inputLayer.getNumNeurons(); y++) {
			for (int x = 0; x < config.inputWidth
					&& x < inputLayer.getNumNeurons(); x++) {
				inputLayer.getNeuron(y * config.inputWidth + x).setValue(
						values[x][y]);
			}
		}
	}

	/**
	 * 
	 */
	public double getInput(int i) {
		if (i >= 0 && i < inputLayer.getNumNeurons()) {
			return inputLayer.getNeuron(i).getValue();
		}
		return Double.MAX_VALUE;
	}

	/**
	 * 
	 */
	public double[] getInputs() {
		double[] inputs = new double[inputLayer.getNeurons().length];
		for (int i = 0; i < inputs.length; i++) {
			inputs[i] = inputLayer.getNeuron(i).getValue();
		}
		return inputs;
	}

	/**
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public double getInput(int x, int y) {
		if (x >= 0 && x < config.inputWidth && x < inputLayer.getNumNeurons()
				&& y >= 0 && y < config.inputHeight
				&& y < inputLayer.getNumNeurons()) {
			return inputLayer.getNeuron(y * config.inputWidth + x).getValue();
		}
		return Double.MAX_VALUE; // エラー: ERROR
	}

	/**
	 * 
	 * @return
	 */
	public double[][] getInputsXY() {
		double[][] inputs = new double[config.inputWidth][config.inputHeight];
		for (int y = 0; y < config.inputHeight
				&& y < inputLayer.getNeurons().length; y++) {
			for (int x = 0; x < config.inputWidth
					&& x < inputLayer.getNeurons().length; x++) {
				// System.out.println("X "+x+ " Y "+y + " OW "+config.inputWidth
				// + " OH "+config.inputHeight);
				inputs[x][y] = inputLayer.getNeuron(y * config.inputWidth + x)
						.getValue();
			}
		}
		return inputs;
	}

	/**
	 * 出力層への一つの出力を得る: get a value from the output layer
	 * 
	 * @param i
	 *            ノード番号: node number
	 * @param value
	 *            値: value
	 */
	public double getOutput(int i) {
		if (i >= 0 && i < outputLayer.getNumNeurons()) {
			return outputLayer.getNeuron(i).getValue();
		}
		return Double.MAX_VALUE; // エラー: ERROR
	}

	/**
	 * 出力層への各出力を得る: get all output values
	 * 
	 * @return values 値: values
	 */
	public double[] getOutputs() {
		double[] outputs = new double[outputLayer.getNeurons().length];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = outputLayer.getNeuron(i).getValue();
		}
		return outputs;
	}

	/**
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public double getOutput(int x, int y) {
		if (x >= 0 && x < config.outputWidth && x < outputLayer.getNumNeurons()
				&& y >= 0 && y < config.outputHeight
				&& y < outputLayer.getNumNeurons()) {
			return outputLayer.getNeuron(y * config.outputWidth + x).getValue();
		}
		return Double.MAX_VALUE; // エラー: ERROR
	}

	/**
	 * 
	 * @return
	 */
	public double[][] getOutputsXY() {
		double[][] outputs = new double[config.outputWidth][config.outputHeight];
		for (int y = 0; y < config.outputHeight
				&& y < outputLayer.getNeurons().length; y++) {
			for (int x = 0; x < config.outputWidth
					&& x < outputLayer.getNeurons().length; x++) {
				// System.out.println("X "+x+ " Y "+y +
				// " OW "+config.outputWidth + " OH "+config.outputHeight);
				outputs[x][y] = outputLayer.getNeuron(
						y * config.outputWidth + x).getValue();
			}
		}
		return outputs;
	}

	/**
	 * 出力層の教師信号を設定する: set the teacher signal for the output layer
	 * 
	 * @param i
	 *            ノード番号: node number
	 * @param value
	 *            教師信号の値: teacher signal value
	 */
	public void setTeacherSignal(int i, double value) {
		if (i >= 0 && i < outputLayer.getNumNeurons()) {
			outputLayer.setTeacherSignal(i, value);
		}
	}

	/**
	 * 出力層の教師信号を設定する: set teacher signal values in the output layer
	 * 
	 * @param values
	 *            全ての教師信号の値: all of the teacher signal values
	 */
	public void setTeacherSignals(double[] values) {
		if (outputLayer.getTeacherSignals().length == values.length) {
			outputLayer.setTeacherSignals(values);
		}
	}

	/**
	 * 出力層の教師信号を設定する: set teacher signal values in the output layer
	 * 
	 * @param values
	 *            全ての教師信号の値: all of the teacher signal values
	 */
	public void setTeacherSignals(double[][] values) {
		for (int y = 0; y < config.outputHeight
				&& y < outputLayer.getNumNeurons(); y++) {
			for (int x = 0; x < config.outputWidth
					&& x < outputLayer.getNumNeurons(); x++) {
				outputLayer.setTeacherSignal(y * config.outputWidth + x,
						values[x][y]);
			}
		}
	}

	/**
	 * 学習率を設定する: set the learning rate
	 * 
	 * @param rate
	 *            学習率: learning rate
	 */
	public void setLearningRate(double rate) {
		config.learningRate = rate;
		inputLayer.setLearningRate(rate);
		for (int i = 0; i < centerLayers.length; i++) {
			centerLayers[i].setLearningRate(rate);
		}
		outputLayer.setLearningRate(rate);
	}

	/**
	 * setters and getter methods
	 */
	public Layer getInputLayer() {
		return inputLayer;
	}

	public Layer[] getCenterLayers() {
		return centerLayers;
	}

	public Layer getOutputLayer() {
		return outputLayer;
	}

	public void setInputLayer(Layer in) {
		inputLayer = in;
	}

	public void setCenterLayers(Layer[] cl) {
		centerLayers = cl;
	}

	public void setCenterLayer(int i, Layer cl) {
		centerLayers[i] = cl;
	}

	public void setOutputLayer(Layer ol) {
		outputLayer = ol;
	}

	public void setInputWidth(int width) {
		config.inputWidth = width;
	}

	public void setInputHeight(int height) {
		config.inputHeight = height;
	}

	public void setOutputWidth(int width) {
		config.outputWidth = width;
	}

	public void setOutputHeight(int height) {
		config.outputHeight = height;
	}

	public int getInputWidth() {
		return config.inputWidth;
	}

	public int getInputHeight() {
		return config.inputHeight;
	}

	public int getOutputWidth() {
		return config.outputWidth;
	}

	public int getOutputHeight() {
		return config.outputHeight;
	}

	public boolean isUseBias() {
		return config.bias;
	}

	public double getLearningRate() {
		return config.learningRate;
	}

}
