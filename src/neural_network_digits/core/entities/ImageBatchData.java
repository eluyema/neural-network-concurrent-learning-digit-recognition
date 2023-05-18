package neural_network_digits.core.entities;

public class ImageBatchData {
	
	private double[] inputBatch;
	private double[] expectedBatch;

	public double[] getInputBatch() {
		return inputBatch;
	}

	public void setInputBatch(double[] inputBatch) {
		this.inputBatch = inputBatch;
	}

	public double[] getExpectedBatch() {
		return expectedBatch;
	}

	public void setExpectedBatch(double[] expectedBatch) {
		this.expectedBatch = expectedBatch;
	}
}
