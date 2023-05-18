package neural_network_digits.core.matrix;

import java.util.Random;

public class Matrix{
	private int rows;
	private int cols;
	
	public interface MatrixConsumer {
		void consume(double value, int row, int col, int index);
	}

	public interface MatrixProducer {
		double produce(double value, int row, int col, int index);
	}

	private double[] a;

	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		a = new double[rows * cols];
	}
	
	public Matrix(int rows, int cols, MatrixFill matrixFill) {
		this(rows, cols);
		for (int i = 0; i < a.length; i++) {
			switch(matrixFill) {
				case GAUSSIAN:
					Random random = new Random();
					a[i] = random.nextGaussian();
					break;
			}

		}
	}
	
	public Matrix(int rows, int cols, double[] values) {
		this.rows = rows;
		this.cols = cols;
		Matrix tmp = new Matrix(cols, rows);
		tmp.a = values;
		Matrix transposed = tmp.transpose();
		a = transposed.a;	
	}
	
	public Matrix map(MatrixProducer producer) {
		Matrix result = new Matrix(rows, cols);
		int index = 0;
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {

				result.a[index] = producer.produce(a[index], row, col, index);

				++index;
			}
		}
		return result;
	}
	
	public Matrix modify(MatrixProducer producer) {
		int index = 0;
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				a[index] = producer.produce(a[index], row, col, index);
				index++;
			}
		}

		return this;
	}
	
	public void forEach(MatrixConsumer consumer) {

		int index = 0;

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				consumer.consume(a[index], row, col, index);
				index++;
			}
		}
	}

	public Matrix multiply(Matrix m) {
		Matrix result = new Matrix(rows, m.cols);

		for (int row = 0; row < result.rows; row++) {
			for (int n = 0; n < cols; n++) {
				for (int col = 0; col < result.cols; col++) {
					result.a[row * result.cols + col] += a[row * cols + n] * m.a[col + n * m.cols];
				}
			}
		}

		return result;
	}
	
	public double sum() {
		double sum = 0;
		
		for(var v: a) {
			sum += v;
		}
		
		return sum;
	}
	
	public Matrix findGreatestRowNumbers() {
	    Matrix result = new Matrix(1, cols);

	    double[] greatestValues = new double[cols];

	    for (int i = 0; i < cols; i++) {
	        greatestValues[i] = Double.MIN_VALUE;
	    }

	    forEach((value, row, col, index) -> {
	        if (value > greatestValues[col]) {
	            greatestValues[col] = value;
	            result.a[col] = row;
	        }
	    });

	    return result;
	}

	public Matrix sumColumns() {
		Matrix result = new Matrix(1, cols);
		
		int index = 0;

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result.a[col] += a[index++];
			}
		}

		return result;
	}
	
	public Matrix transpose() {
		Matrix result = new Matrix(cols, rows);
		
		for(int i = 0; i < a.length; ++i) {
			int row = i / cols;
			int col = i % cols;
			
			result.a[col * rows + row] = a[i];
		}
		
		return result;
	}
	
	public Matrix averageColumn() {
		Matrix result = new Matrix(rows, 1);
		
		forEach((value, row, col, index)->{
			result.a[row] += value/cols;
		});
		
		return result;
	}
	
	public Matrix exponent() {
		Matrix result = new Matrix(rows, cols);
				
		for(int i = 0; i < rows*cols; i++) {
			result.a[i] = Math.exp(a[i]);
		}
		
		return result;
	}
	
	public Matrix softmax() {
		Matrix result = exponent();
		
		Matrix colSum = result.sumColumns();
		
		result.modify((value, row, col, index)->{
			return value/colSum.getByIndex(col);
		});
		
		return result;
	}
	
	
	public Matrix addIncrement(int row, int col, double increment) {
		
		Matrix result = map((value, rowi, coli, index)->a[index]);
		
		double originalValue = get(row, col);
		
		double newValue = originalValue + increment;
		
		result.set(row, col, newValue);
		
		return result;
	}

	public int getRows() {
		return rows;
	}
	
	public int getCols() {
		return cols;
	}

	public double getByIndex(int index) {
		return a[index];
	}

	public double[] get() {
		return a;
	}

	public void set(int row, int col, double value) {
		a[row * cols + col] = value;
	}
	
	public double get(int row, int col) {
		return a[row * cols + col];
	}
}
