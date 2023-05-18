package neural_network_digits.core.learningAlgorithms;

import java.util.LinkedList;

import neural_network_digits.core.engine.NeuralNetwork;
import neural_network_digits.core.entities.ImageBatchData;
import neural_network_digits.core.entities.ImageDataInfo;
import neural_network_digits.core.entities.NeuralNetworkResult;
import neural_network_digits.core.managers.TrainDataManager;
import neural_network_digits.core.matrix.Matrix;

public class SequentialAlgorithm {
	private double learningRate;
	private int epochsSize;
	private NeuralNetwork neuralNetwork;

	public SequentialAlgorithm(double learningRate, int epochsSize, NeuralNetwork neuralNetwork) {
		this.learningRate = learningRate;
		this.epochsSize = epochsSize;
		this.neuralNetwork = neuralNetwork;
	}
	
	public void train(TrainDataManager trainDataManager, TrainDataManager examDataManager) throws Exception {
		for (int epoch = 0; epoch < epochsSize; epoch++) {

			System.out.println("E" + epoch);

			executeEpoch(trainDataManager, true);
			
			executeEpoch(examDataManager, false);


			learningRate -= (learningRate * 0.85) / epochsSize;
		}
	}

	private void executeEpoch(TrainDataManager examDataManager, boolean trainingMode) throws Exception {

		examDataManager.open();

		LinkedList<NeuralNetworkResult> batchTasks = generateBatchTasks(examDataManager, trainingMode);
		processBatchResults(batchTasks, trainingMode);

		examDataManager.close();
	}

	private void processBatchResults(LinkedList<NeuralNetworkResult> batchesResult, boolean trainingMode) throws Exception {
		double avgAccuracy = 0;

		for(NeuralNetworkResult batchResult: batchesResult) {
			try {
				if(!trainingMode) {
					avgAccuracy += batchResult.getTrainingAccuracy();
				}
			} catch (Exception e) {
				throw new Exception("Error appeared while ");
			} 
		}
		
		if(!trainingMode) {
			avgAccuracy = avgAccuracy / batchesResult.size();
			String accuracyStr = String.format("%.2f", avgAccuracy);
			System.out.println("\nAccuracy = " + accuracyStr + "%\n");
		}

	}

	private LinkedList<NeuralNetworkResult> generateBatchTasks(TrainDataManager dataManager, boolean trainingMode) throws Exception {
		
		LinkedList<NeuralNetworkResult> batchTasks = new LinkedList<>();

		ImageDataInfo imageDataInfo = dataManager.getImageDataInfo();
		int batchesSize = imageDataInfo.getNumberBatches();
		int processStep = batchesSize / 16;
		for (int index = 0; index < batchesSize; index++) {
			ImageBatchData batchData = dataManager.readBatch();
			
			batchTasks.add(executeBatch(imageDataInfo, batchData, trainingMode));
			
			if(trainingMode && index != 0 && index % processStep == 0) {
				System.out.print("|");
			}
		}

		return batchTasks;
	}

	private NeuralNetworkResult executeBatch(ImageDataInfo imageDataInfo, ImageBatchData batchData, boolean trainingMode) throws Exception {


		int itemsRead = imageDataInfo.getItemsRead();

		int inputSize = imageDataInfo.getInputSize();
		int expectedSize = imageDataInfo.getExpectedSize();

		Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
		Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());
		
		NeuralNetworkResult nnResult = neuralNetwork.executeForwardPropagation(input);
		
		if(trainingMode) {
			neuralNetwork.executeBackwardPropagation(nnResult, expected);;
			
			neuralNetwork.updateParameters(nnResult, learningRate);
		}
		else {
			neuralNetwork.evaluateNeuralNetworkPerformance(nnResult, expected);
		}
		
		return nnResult;

	}
}
