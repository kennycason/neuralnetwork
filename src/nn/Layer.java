package nn;

import java.util.Random;

/**
 * ニューラルネット層のクラス: Neural Network class
 * @author kenneth cason
 */
public class Layer {
	
	private int numNeurons;	// 神経数: number of neurons
	
	private double[] teacherSignals;// 教師信号: teacher signal
	
	private double[] errors;		// 誤差: error
	
	private double learningRate;	// 学習率: learning rate
	
	private Neuron[] neurons;
	
	private boolean useBias = false;
	
	private double[] biasValues; 	// バイアス値（閾値）: bias value bias value 
	
	private double[] biasWeights; 	// バイアスの重み:bias weights	
	
	Layer parentLayer;		// 親層: parent layer
	
	Layer childLayer;		// 子層: child layer

	/*
	 * 層を初期化する時、重みをランダムに設定する
	 * Set weights randomly upon initialization of the layer
	 */
	Random rand;
	
	public Layer() {
		parentLayer = null;
		childLayer = null;
		rand = new Random();
	}
	
	/**
	 * 層を初期化する: initialize the layer
	 * @param numNeurons　この層の神経数: the number of neurons in this layer
	 * @param parent　親層: parent layer
	 * @param child　子層: child layer
	 */
	public void init(int _numNeurons, Layer parent, Layer child, boolean bias) {
		useBias = bias;
		numNeurons = _numNeurons;
		neurons = new Neuron[numNeurons];
		teacherSignals = new double[numNeurons];
		errors = new double[numNeurons];
	    for(int i = 0; i < numNeurons; i++) {
	    	neurons[i] = new Neuron(); // init all values
	    }
	    
		if (parent != null) {
			parentLayer = parent;
		}
		if (child != null) {
			childLayer = child;
			// connect each node to each node in the child layer
		    for(Neuron node : neurons) {

		    	for(int i = 0; i < childLayer.numNeurons; i++) {
		    		node.connectNode(childLayer.neurons[i]);
		    		// 重みとバイアス重みを初期化する: initialize the weights
		    		node.setWeight(i, rand.nextInt(200) / 100.0 - 1);
		    	}
		    }
		    if(useBias) {
		    	biasValues = new double[childLayer.numNeurons];
		    	biasWeights = new double[childLayer.numNeurons];
				// バイアスも-1.0〜1.0
				for (int i = 0; i < biasWeights.length; i++) {
					biasWeights[i] = rand.nextInt(200) / 100.0 - 1;
				}
				for(int i = 0; i < biasValues.length; i++) {
					biasValues[i] = -1;
				}
		    }
		} else {
			 for(Neuron node : neurons) {
				 node.setWeights(null);
			 }
			 biasValues = null;
		}
		
		for (int i = 0; i < numNeurons; i++) {
			neurons[i].setValue(0);
			teacherSignals[i] = 0;
			errors[i] = 0;
		}
		if(childLayer != null) {

			for(Neuron node : neurons) {
	    		for(int i = 0; i < node.getAllLinked().size(); i++) {
	    			// 重みとバイアス重みを初期化する: initialize the weights 
	    			node.setWeight(i, rand.nextInt(200) / 100.0 - 1); 
	    		}
	    	}
		}
	}
	
	/**
	 * 誤差を計算する
	 * calculate the error
	 */
	public void calculateErrors() {
		if (childLayer == null) { // 出力層: output layer
			for (int i = 0; i < numNeurons; i++) {
				errors[i] = (teacherSignals[i] - neurons[i].getValue()) * neurons[i].getValue() * (1.0 - neurons[i].getValue());
			}
		} else if (parentLayer == null) { // 入力層: input layer
			for (int i = 0; i < numNeurons; i++) {
				errors[i] = 0.0;
			}
		} else { // 中間層: middle layer
			for (int i = 0; i < numNeurons; i++) {
				double sum = 0;
				for (int j = 0; j < neurons[i].getAllLinked().size(); j++) {
					sum += childLayer.errors[j] * neurons[i].getWeight(j);
				}
				errors[i] = sum * neurons[i].getValue() * (1.0 - neurons[i].getValue());
			}
		}
	}
	
	/**
	 * 誤差によると、結合荷重を調整する
	 * depending on the error, adjust the weights
	 */
	public void adjustWeights() {
		if (childLayer != null) {
			// 重みを調整する: adjust the wegihts
			for (int i = 0; i < numNeurons; i++) {
				for (int j = 0; j < neurons[i].getAllLinked().size(); j++) {
					neurons[i].setWeight(j, neurons[i].getWeight(j)
							+ learningRate * childLayer.errors[j] * neurons[i].getValue());
				}
			}

			if(useBias) {
				for(int i = 0; i < childLayer.numNeurons; i++) {
					biasWeights[i] += learningRate * childLayer.errors[i] * biasValues[i];
				}
			}
		}
	}
	
	/**
	 * この層の各ニューロンの値をゼロにする
	 * clear each layer's neuron values
	 */
	public void clearAllValues() {
		for (int i = 0; i < numNeurons; i++) {
			neurons[i].setValue(0);
		}
	}
	
	/**
	 * この層の各ニューロンの活性値を計算する
	 * calculate the neuron value for every neuron in this layer
	 */
	public void calculateNeuronValues() {
		double sum;
		if (childLayer != null) {
			for (Neuron thisNode : neurons) {
				// dont need to run values of input layer through function
				if(parentLayer != null) {
					thisNode.setValue(sigmoid(thisNode.getValue()));
				}
				for (int i = 0; i < thisNode.getAllLinked().size(); i++) {
						sum = 0.0;
						sum += thisNode.getValue() * thisNode.getWeight(i);
						if(useBias) {
							sum += biasValues[i] * biasWeights[i];
						}
						thisNode.getAllLinked().get(i).setValue(thisNode.getAllLinked().get(i).getValue() + sum);

				}	
			} 
		} else { 
			// is the output layer, so just run the values through the sigmoid function
			for (Neuron thisNode : neurons) {
				thisNode.setValue(sigmoid(thisNode.getValue()));
			}
		}
	}
	
	/**
	 * シグモイド関数: Sigmoid function
	 */
	private double sigmoid(double x) {
		return (1.0 / (1 + Math.exp(-x)));
	}
	
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public Neuron getNeuron(int i) {
		return neurons[i];
	}
	
	public int getNumNeurons() {
		return numNeurons;
	}
	
	public boolean isUseBias() {
		return useBias;
	}
	
	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double rate) {
		learningRate = rate;
	}
	
	public double getNeuronLearningRateCoefficient(int i) {
		return neurons[i].getLearningRateCoefficient();
	}
	
	public void setNeuronLearningRateCoefficient(int i, double rate) {
		neurons[i].setLearningRateCoefficient(rate);
	}
	
	public double getTeacherSignal(int i) {
		return teacherSignals[i];
	}
	
	public double[] getTeacherSignals() {
		return teacherSignals;
	}
	
	public void setTeacherSignals(double[] signals) {
		teacherSignals = signals;
	}
	
	public void setTeacherSignal(int i, double signal) {
		teacherSignals[i] = signal;
	}

}
