package neural_network_digits.core.managers;

import java.io.File;

import neural_network_digits.core.engine.ActivationFunction;
import neural_network_digits.core.engine.NetworkEngine;
import neural_network_digits.core.engine.NeuralNetwork;
import neural_network_digits.core.entities.ImageDataInfo;
import neural_network_digits.core.learningAlgorithms.*;
import neural_network_digits.core.matrix.Matrix;

public class NeuralNetworkManager {
	private String regularImagesLocation = null;
	private String regularLabelsLocation = null;
	private String examImagesLocation = null;
	private String examLabelsLocation = null;
	
	private int oneBatchImagesCount = 32;
	
	private ImageDataInfo imagesDataInfo = null;
	
	private TrainDataManager regularDataManager = null;
	private TrainDataManager examDataManager = null;
	
	private NeuralNetwork neuralNetwork = null;
	
	public NeuralNetworkManager(String trainDataDirectory, int hiddenLayerNeurons) throws Exception {
		setTrainDataLocation(trainDataDirectory);
		regularDataManager.open();
		this.imagesDataInfo = regularDataManager.getImageDataInfo();
		regularDataManager.close();
		
		initNeuralNetwork(hiddenLayerNeurons);
	}
	
	private void setTrainDataLocation(String trainDataDirectory) throws Exception  {
		File directory = new File(trainDataDirectory);

		if (!directory.isDirectory()) {
			throw new Exception("Directory doesn't exist");
		}
	
		this.regularImagesLocation = String.format("%s%s%s", trainDataDirectory, File.separator, "train-images-idx3-ubyte");
		this.regularLabelsLocation = String.format("%s%s%s", trainDataDirectory, File.separator, "train-labels-idx1-ubyte");
		this.examImagesLocation = String.format("%s%s%s", trainDataDirectory, File.separator, "t10k-images-idx3-ubyte");
		this.examLabelsLocation = String.format("%s%s%s", trainDataDirectory, File.separator, "t10k-labels-idx1-ubyte");
		
		this.regularDataManager = new TrainDataManager(regularImagesLocation, regularLabelsLocation, oneBatchImagesCount);
		this.examDataManager = new TrainDataManager(examImagesLocation, examLabelsLocation, oneBatchImagesCount);
	}
	
	private void initNeuralNetwork(int hiddenLayerNeurons) {
		neuralNetwork = null;
		
		neuralNetwork = new NeuralNetwork();
		
		int firstLayerSize = this.imagesDataInfo.getInputSize();
		int lastLayerSize = this.imagesDataInfo.getExpectedSize();
		
		neuralNetwork.addLayer(ActivationFunction.LINEAR, hiddenLayerNeurons, firstLayerSize);
		neuralNetwork.addLayer(ActivationFunction.RELU);
		neuralNetwork.addLayer(ActivationFunction.LINEAR, lastLayerSize);
		neuralNetwork.addLayer(ActivationFunction.SOFTMAX);
	}
	
	public void startTrainInParallel(int epochsSize, double learningRate, int threads) throws Exception {
		ParallelAlgorithm alghorithm = new ParallelAlgorithm(learningRate, epochsSize, neuralNetwork, threads);
		alghorithm.train(regularDataManager, examDataManager);
	}
	
	public void startTrainInSequential(int epochsSize, double learningRate) throws Exception {
		SequentialAlgorithm alghorithm = new SequentialAlgorithm(learningRate, epochsSize, neuralNetwork);
		alghorithm.train(regularDataManager, examDataManager);
	}
	
 	public int getPredictedNumber(Matrix input) {
 		Matrix result = neuralNetwork.executeForwardPropagation(input).getOutput();
 		
		return (int)result.findGreatestRowNumbers().get(0, 0);
	}
 	
}
