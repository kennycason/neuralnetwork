package nn;

public class NeuralNetworkConfig {
	
	/**
	 * 入力層のノード数: number of nodes in the input layer
	 */
	public int numInputNodes = 1;
	
	/**
	 * 中間層数: number of center layers
	 */
	public int numCenterLayers = 1;
	
	/**
	 * 中間層のノード数: number of nodes in the center layer
	 */
	public int numCenterNodes = 10;
	
	/**
	 * 出力層のノード数: number of nodes in the output layer
	 */
	public int numOutputNodes = 1;
	
	public boolean bias = false;
	
	/**
	 * 
	 */
	public double learningRate = 0.1;
	
	public int inputWidth = 1;
	
	public int inputHeight = 1;
	
	public int outputWidth = 1;
	
	public int outputHeight = 1;

}
