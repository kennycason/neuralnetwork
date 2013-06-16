package nn;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.junit.Test;

public class ChessImageTest {

	@Test
	public void Test() {

		double[][] trainData = {
				loadImageData("nn/chess/black_king.png"),
				loadImageData("nn/chess/black_queen.png"),
				loadImageData("nn/chess/black_rook.png"),
				loadImageData("nn/chess/black_knight.png"),
				loadImageData("nn/chess/black_bishop.png"),
				loadImageData("nn/chess/black_pawn.png"),
				
				loadImageData("nn/chess/white_king.png"),
				loadImageData("nn/chess/white_queen.png"),
				loadImageData("nn/chess/white_rook.png"),
				loadImageData("nn/chess/white_knight.png"),
				loadImageData("nn/chess/white_bishop.png"),
				loadImageData("nn/chess/white_pawn.png")
		};
		
		/*
		 * first two digits represent color, black, then white, 
		 * note, use separate bits for colors for faster learning
		 */
		double[][] trainResults = {
				{1, 0, 1, 0, 0, 0, 0, 0}, // BK
				{1, 0, 0, 1, 0, 0, 0, 0}, // BQ
				{1, 0, 0, 0, 1, 0, 0, 0}, // BR
				{1, 0, 0, 0, 0, 1, 0, 0}, // BKn
				{1, 0, 0, 0, 0, 0, 1, 0}, // BB
				{1, 0, 0, 0, 0, 0, 0, 1}, // BP
				
				{0, 1, 1, 0, 0, 0, 0, 0}, // WK
				{0, 1, 0, 1, 0, 0, 0, 0}, // WQ
				{0, 1, 0, 0, 1, 0, 0, 0}, // WR
				{0, 1, 0, 0, 0, 1, 0, 0}, // WKn
				{0, 1, 0, 0, 0, 0, 1, 0}, // WB
				{0, 1, 0, 0, 0, 0, 0, 1}, // WP
		};
		
		NeuralNetworkConfig config = new NeuralNetworkConfig();
		config.numInputNodes = trainData[0].length;
		config.numCenterLayers = 2; 
		config.numCenterNodes = 15;
		config.numOutputNodes = trainResults[0].length;
		config.bias = false;
		config.learningRate = 0.07;
		
		NeuralNetwork nn = new NeuralNetwork(config);
		
		run(nn, trainData, trainResults);
		System.out.println("Data successfully trained :)");
		System.out.println("messy data: test NN's ability to recognize new data");
		testNN(nn, loadImageData("nn/chess/black_king_messy.png"));
		testNN(nn, loadImageData("nn/chess/white_queen_messy.png"));
		System.out.println("failed to recognize :(");
		
	}
	
	private double[] loadImageData(String file) {
		double[] data = null;
		try {
			BufferedImage originalImage = ImageIO.read(Thread.currentThread().getContextClassLoader().getResource(file));
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			ImageIO.write(originalImage, "png", baos);
			byte[] imageInByte = baos.toByteArray();
			data = new double[1300];
			for (int i = 0; i < data.length && i < imageInByte.length; i++) {
				data[i] = (imageInByte[i] + 128) / 255.0;
			}
			System.out.println(data.length + ": " + Arrays.toString(data));
		} catch (Exception e) {
			System.err.println("Failed Loading Image: " + file);
		}
		return data;
	}

	private void run(NeuralNetwork nn, double[][] trainData, double[][] trainResults) {
		double maxError = 0.01;
		double error = Double.MAX_VALUE;
		
		int count = 0;
		System.out.println("Begin trainings");
		while(error > maxError) {
			error = 0;
			for(int i = 0; i < trainData.length; i++) {
				for(int j = 0; j < trainData[i].length; j++) {
					nn.setInput(j, trainData[i][j]);
				}
				for(int j = 0; j < trainResults[i].length; j++) {
					nn.setTeacherSignal(j, trainResults[i][j]);
				}
				nn.feedForward();
				error += nn.calculateError();
				nn.backPropagate();
			}
			count++;
			error /= trainData.length;
			if(count % 100 == 0)  {
				System.out.println("[" + count + "] error = " + error);
			}
		}

		for(int i = 0; i < trainData.length; i++) {
			testNN(nn, trainData[i]);
		}
		
	}
	
	private void testNN(NeuralNetwork nn, double[] trainData) {
		for(int j = 0; j < trainData.length; j++) {
			nn.setInput(j, trainData[j]);
		}
		nn.feedForward();
		for(int j = 0; j < nn.getOutputs().length; j++) {
			if(nn.getOutput(j) > 0.4) {
				System.out.print("1 ");
			} else {
				System.out.print("0 ");
			}
		}
		System.out.println();
	}
	
}
