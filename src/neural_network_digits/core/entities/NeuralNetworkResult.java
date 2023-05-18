package neural_network_digits.core.entities;

import java.util.LinkedList;

import neural_network_digits.core.matrix.Matrix;

public class NeuralNetworkResult {
    private LinkedList<Matrix> outputRecords = new LinkedList<>();
    private LinkedList<Matrix> weightErrorRecords = new LinkedList<>();
    private LinkedList<Matrix> weightInputRecords = new LinkedList<>();
    
    private Matrix inputErrorMatrix;
    private double trainingLoss;
    private double trainingAccuracy;
    
    public void addWeightInput(Matrix weightInput) {
        weightInputRecords.add(weightInput);
    }
    
    public LinkedList<Matrix> getWeightInputs() {
        return weightInputRecords;
    }
    
    public LinkedList<Matrix> getOutputRecords() {
        return outputRecords;
    }
    
    public void addOutputRecord(Matrix output) {
        outputRecords.add(output);
    }
    
    public Matrix getOutput() {
        return outputRecords.get(outputRecords.size() - 1);
    }

    public LinkedList<Matrix> getWeightErrorRecords() {
        return weightErrorRecords;
    }

    public void addWeightErrorRecord(Matrix weightError) {
        weightErrorRecords.add(0, weightError);
    }

    public Matrix getInputErrorMatrix() {
        return inputErrorMatrix;
    }

    public void setInputErrorMatrix(Matrix inputError) {
        this.inputErrorMatrix = inputError;
    }

    public void setTrainingLoss(double loss) {
        this.trainingLoss = loss;
    }
    
    public double getTrainingLoss() {
        return trainingLoss;
    }

    public void setTrainingAccuracy(double accuracy) {
        this.trainingAccuracy = accuracy;
    }
    
    public double getTrainingAccuracy() {
        return trainingAccuracy;
    }
}
