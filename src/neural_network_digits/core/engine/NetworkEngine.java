package neural_network_digits.core.engine;

import java.util.LinkedList;

import neural_network_digits.core.entities.NeuralNetworkResult;
import neural_network_digits.core.matrix.Matrix;
import neural_network_digits.core.matrix.MatrixFill;


public class NetworkEngine {
	private LinkedList<ActivationFunction> activationFunctions = new LinkedList<>();
	private LinkedList<Matrix> weights = new LinkedList<>();
	private LinkedList<Matrix> biases = new LinkedList<>();
	
	private boolean shouldStoreInputError = false;
	
public NeuralNetworkResult runForwards(Matrix input) {
		
		NeuralNetworkResult nnResult = new NeuralNetworkResult();
		Matrix output = input;
		int weightedLayerIndex = 0;
		
		nnResult.addOutputRecord(output);
		
		for(var t: activationFunctions) {
			if(t == ActivationFunction.LINEAR) {
				
				nnResult.addWeightInput(output);
				Matrix weight = weights.get(weightedLayerIndex);
				Matrix bias = biases.get(weightedLayerIndex);
				
				output = weight.multiply(output).modify((value, row, col, index) -> value + bias.getByIndex(row));
				
				weightedLayerIndex++;
			}
			else if(t == ActivationFunction.RELU) {
				output = output.modify((value, row, col, index) -> value > 0 ? value: 0);
			}
			else if(t == ActivationFunction.SOFTMAX) {
				output = output.softmax();
			}
			
			nnResult.addOutputRecord(output);
		}

		return nnResult;
	}
	
	public void updateParameters(NeuralNetworkResult nnResult, double learningRate) {
		var weightInputs = nnResult.getWeightInputs();
		var weightErrors = nnResult.getWeightErrorRecords();
		
		for(int i = 0; i < weights.size(); i++) {
			var weight = weights.get(i);
			var bias = biases.get(i);
			var error = weightErrors.get(i);
			var input = weightInputs.get(i);
			
			
			var weightAdjust = error.multiply(input.transpose());
			var biasAdjust = error.averageColumn();
			
			double rate = learningRate/input.getCols();
			
			weight.modify((value, row, col, index)->value - rate * weightAdjust.getByIndex(index));
			bias.modify((value, row, col, index)->value - learningRate * biasAdjust.getByIndex(row));
		}
		
	}
	
	public void backwardPropagation(NeuralNetworkResult nnResult, Matrix expected) {
		
		var transformOperationIt = activationFunctions.descendingIterator();
		
		var ioIt = nnResult.getOutputRecords().descendingIterator();
		var weightIt = weights.descendingIterator();
		Matrix softmaxOutput = ioIt.next();
		Matrix error = softmaxOutput.map((value, row, col, index)->value - expected.getByIndex(index));
		
		while(transformOperationIt.hasNext()) {
			ActivationFunction transform = transformOperationIt.next();
			
			Matrix input = ioIt.next();
			
			switch(transform) {
			case LINEAR:
				Matrix weight = weightIt.next();
				
				nnResult.addWeightErrorRecord(error);
				
				if(weightIt.hasNext() || shouldStoreInputError) {
					error = weight.transpose().multiply(error);
				}
				break;
			case RELU:
				error = error.map((value, row, col, index)->input.getByIndex(index) > 0 ? value: 0);
				break;
			case SOFTMAX:
				break;
			}
		}
		
		if(shouldStoreInputError) {
			nnResult.setInputErrorMatrix(error);
		}
	}

	public void setStoreInputError(boolean shouldStoreInputError) {
		this.shouldStoreInputError = shouldStoreInputError;
	}


	public void addLayer(ActivationFunction activationFunction, double... layerArgs) {
	    if (activationFunction == ActivationFunction.LINEAR) {
	        int neuronCount = (int) layerArgs[0];
	        int weightsSize = weights.isEmpty() ? (int) layerArgs[1] : weights.getLast().getRows();
	        Matrix weight = new Matrix(neuronCount, weightsSize, MatrixFill.GAUSSIAN);
	        Matrix bias = new Matrix(neuronCount, 1, MatrixFill.ZERO);
	        
	        weights.add(weight);
	        biases.add(bias);
	    }
	    
	    activationFunctions.add(activationFunction);
	}
	
	public void evaluateNeuralNetworkPerformance(NeuralNetworkResult nnResult, Matrix expected) {
	    double loss = crossEntropy(expected, nnResult.getOutput()).averageColumn().getByIndex(0);
	    Matrix prognostication = nnResult.getOutput().findGreatestRowNumbers();
	    Matrix real = expected.findGreatestRowNumbers();
	    int accurate = 0;

	    for (int i = 0; i < real.getCols(); i++) {
	        int realValue = (int) real.getByIndex(i);
	        int prognosticationValue = (int) prognostication.getByIndex(i);
	        if (realValue == prognosticationValue) {
	            accurate++;
	        }
	    }

	    int all = real.getCols();
	    double accuracy = (100.0 * accurate) / all;

	    nnResult.setTrainingLoss(loss);
	    nnResult.setTrainingAccuracy(accuracy);
	}
	
	private Matrix crossEntropy(Matrix expected, Matrix actual) {	
		return actual.map((value, row, col, index)->{
			return Math.log(value) * (-expected.getByIndex(index));
		}).sumColumns();
	}
}
