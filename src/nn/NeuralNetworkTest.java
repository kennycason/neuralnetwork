package nn;

import org.junit.Test;

public class NeuralNetworkTest {

	@Test
	public void test() {
		NeuralNetworkConfig config = new NeuralNetworkConfig();
		config.bias = false;
		config.numCenterLayers = 1;
		config.numCenterNodes = 10;
		config.numInputNodes = 2;
		config.numOutputNodes = 1;
		config.learningRate = 0.55;
		NeuralNetwork nn = new NeuralNetwork(config);
		
		int numTrainData = 4;
		double[][] trainData = new double[numTrainData][nn.getInputs().length];
		double[][] teacherSignal = new double[numTrainData][nn.getOutputs().length];
		
		trainData[0] = new double[]{0.0, 0.0};  // 0 & 0
		trainData[1] = new double[]{0.0, 1.0};  // 0 & 1
		trainData[2] = new double[]{1.0, 0.0};  // 1 & 0
		trainData[3] = new double[]{1.0, 1.0};  // 1 & 1

		teacherSignal[0] = new double[]{0.0};
		teacherSignal[1] = new double[]{0.0};
		teacherSignal[2] = new double[]{0.0};
		teacherSignal[3] = new double[]{1.0};
		
		// Train
		double maxError = 0.001;
		double error = Double.MAX_VALUE;
		int count = 0;
		System.out.println("Begin trainings");
		while(error > maxError) {
			error = 0;
			for(int i = 0; i < numTrainData; i++) {
				for(int j = 0; j < config.numInputNodes; j++) {
					nn.setInput(j, trainData[i][j]);
					nn.setInput(j, trainData[i][j]);
				}
				for(int j = 0; j < config.numOutputNodes; j++) {
					nn.setTeacherSignal(j, teacherSignal[i][j]);
				}
				nn.feedForward();
				error += nn.calculateError();
				nn.backPropagate();
				nn.clearAllValues();
			}
			count++;
			error /= numTrainData;
			if(count % 100 == 0) {
				System.out.println("[" + count + "] error = " + error);
			}
		}
		
		// print results
		for(int i = 0; i < numTrainData; i++) {
			nn.clearAllValues();
			System.out.print("[ ");
			for(int j = 0; j < nn.getInputs().length; j++) {
				nn.setInput(j, trainData[i][j]);
				System.out.print(" " + trainData[i][j]);
			}
			System.out.print("] => [ ");
			nn.feedForward();
			for(int j = 0; j < nn.getOutputs().length; j++) {
				nn.setTeacherSignal(j, teacherSignal[i][j]);
				System.out.print(nn.getOutput(j));
			}
			System.out.println("]");
		}
	}

}
