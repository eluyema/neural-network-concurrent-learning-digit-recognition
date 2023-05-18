package neural_network_digits.core.engine;

import neural_network_digits.core.entities.NeuralNetworkResult;
import neural_network_digits.core.matrix.Matrix;

public class NeuralNetwork {
	private NetworkEngine networkEngine;

	public NeuralNetwork() {
		networkEngine = new NetworkEngine();
	}

	public void addLayer(ActivationFunction activationFunc, double... params) {
		networkEngine.addLayer(activationFunc, params);
	}
	
	public double[] predict(double[] inputData) {
		
		Matrix input = new Matrix(inputData.length, 1, inputData);
		
		NeuralNetworkResult nnResult = networkEngine.runForwards(input);
		return nnResult.getOutput().get();
	}
	
	public NeuralNetworkResult executeForwardPropagation(Matrix input) {
		return networkEngine.runForwards(input);
	}
	
	public void executeBackwardPropagation(NeuralNetworkResult nnResult, Matrix expectedResult) {
		networkEngine.backwardPropagation(nnResult, expectedResult);
	}
	
	public void updateParameters(NeuralNetworkResult nnResult, double learningRate) {
		networkEngine.updateParameters(nnResult, learningRate);
	}
	
	public void evaluateNeuralNetworkPerformance(NeuralNetworkResult nnResult, Matrix expected) {
		networkEngine.evaluateNeuralNetworkPerformance(nnResult, expected);
	}
}
