package neural_network_digits.core.learningAlgorithms;

import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import neural_network_digits.core.engine.NeuralNetwork;
import neural_network_digits.core.entities.ImageBatchData;
import neural_network_digits.core.entities.ImageDataInfo;
import neural_network_digits.core.entities.NeuralNetworkResult;
import neural_network_digits.core.managers.TrainDataManager;
import neural_network_digits.core.matrix.Matrix;

public class ParallelAlgorithm {
	private double learningRate;
	private int epochsSize;
	private int threads = 4;
	private NeuralNetwork neuralNetwork;
	
	private Lock lock;
	
	public ParallelAlgorithm(double learningRate, int epochsSize, NeuralNetwork neuralNetwork, int threads) {
		this.learningRate = learningRate;
		this.epochsSize = epochsSize;
		this.threads = threads;
		this.neuralNetwork = neuralNetwork;
		
		this.lock = new ReentrantLock();
		
	}
	
	public void train(TrainDataManager trainDataManager, TrainDataManager examDataManager) throws Exception {
		for (int epoch = 0; epoch < epochsSize; epoch++) {

			System.out.println("E" + epoch);

			executeEpoch(trainDataManager, threads, true);
			
			executeEpoch(examDataManager, threads, false);


			learningRate -= (learningRate * 0.85) / epochsSize;
		}
	}

	private void executeEpoch(TrainDataManager examDataManager, int threads, boolean trainingMode) throws Exception {

		examDataManager.open();

		LinkedList<Future<NeuralNetworkResult>> batchTasks = generateBatchTasks(examDataManager, threads, trainingMode);
		processBatchResults(batchTasks, trainingMode);

		examDataManager.close();
	}

	private void processBatchResults(LinkedList<Future<NeuralNetworkResult>> batchesTasks, boolean trainingMode) throws Exception {
		int index = 0;

		double avgAccuracy = 0;
			
		int processStep = batchesTasks.size() / 16;

		for(Future<NeuralNetworkResult> batch: batchesTasks) {
			try {
				NeuralNetworkResult batchResult = batch.get();
				
				if(!trainingMode) {
					avgAccuracy += batchResult.getTrainingAccuracy();
				}
			} catch (Exception e) {
				throw new Exception("Error appeared while ");
			} 
			
			if(trainingMode && index != 0 && index % processStep == 0) {
				System.out.print("|");
			}
			index++;
		}
		
		if(!trainingMode) {
			avgAccuracy = avgAccuracy / batchesTasks.size();
			String accuracyStr = String.format("%.2f", avgAccuracy);
			System.out.println("\nAccuracy = " + accuracyStr + "%\n");
		}

	}

	private LinkedList<Future<NeuralNetworkResult>> generateBatchTasks(TrainDataManager dataManager, int threads, boolean trainingMode) {
		
		LinkedList<Future<NeuralNetworkResult>> batchTasks = new LinkedList<>();

		ImageDataInfo metaData = dataManager.getImageDataInfo();
		int batchesSize = metaData.getNumberBatches();
		
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		int imagesCount = 0;
		for (int i = 0; i < batchesSize; i++) {
			imagesCount+=32;
			ImageBatchData batchData = dataManager.readBatch();
			
			batchTasks.add(executor.submit(()->executeBatch(metaData, batchData, trainingMode)));
		}
		String message = "";
		if (trainingMode) {
			message+= "While training ";
		} else {
			message+= "While examp test ";
		}
		message+= "was read " + imagesCount;
		System.out.println(message);
		executor.shutdown();

		return batchTasks;
	}

	private NeuralNetworkResult executeBatch(ImageDataInfo metaData, ImageBatchData batchData, boolean trainingMode) throws Exception {


		int itemsRead = metaData.getItemsRead();

		int inputSize = metaData.getInputSize();
		int expectedSize = metaData.getExpectedSize();

		Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
		Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());
		
		NeuralNetworkResult nnResult = neuralNetwork.executeForwardPropagation(input);
		
		if(trainingMode) {
			neuralNetwork.executeBackwardPropagation(nnResult, expected);;
			
			lock.lock();
			try {
				neuralNetwork.updateParameters(nnResult, learningRate);
			} finally {
				lock.unlock();
			}
		}
		else {
			neuralNetwork.evaluateNeuralNetworkPerformance(nnResult, expected);
		}
		
		return nnResult;

	}
}
