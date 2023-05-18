package neural_network_digits;

import neural_network_digits.core.managers.NeuralNetworkManager;
import neural_network_digits.test.NeuralNetworkTest;

public class NeuralNetworkProgram {

  public static void main(String[] args) {

    String directoryTrainData = "./data/MNIST";
    String directoryTestImages = "C:\\Users\\danylo\\eclipse-workspace\\neural_network_digits\\neural_network_digits\\test_arena";
    
    int threads = 4;
    
    boolean runInParallel = true;
    int hiddenLayerNeurons = 300;
    int epochsSize = 1;
    double learningRate = 0.01;
    
    try {
	      NeuralNetworkManager manager = new NeuralNetworkManager(directoryTrainData, hiddenLayerNeurons);
	      long startTime = System.currentTimeMillis();
	      if(runInParallel) {
		      manager.startTrainInParallel(epochsSize, learningRate, threads);
	      } else {
	    	  manager.startTrainInSequential(epochsSize, learningRate);
	      }
	      long endTime = System.currentTimeMillis();
	      long elapsedTime = endTime - startTime;
	      System.out.println("Elapsed time: " + elapsedTime + " milliseconds");

		  NeuralNetworkTest neuralNetworkTest = new NeuralNetworkTest(directoryTestImages, manager);
		  neuralNetworkTest.test();
    
    } catch (Exception e) {
        System.out.println("Error appearead!");
        e.printStackTrace();
      }
  }

}