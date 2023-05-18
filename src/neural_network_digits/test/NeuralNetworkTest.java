package neural_network_digits.test;

import javax.imageio.ImageIO;

import neural_network_digits.core.entities.ImageBatchData;
import neural_network_digits.core.managers.NeuralNetworkManager;
import neural_network_digits.core.matrix.Matrix;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class NeuralNetworkTest {
	private String dirname;
	private NeuralNetworkManager neuralNetworkManager;
	public NeuralNetworkTest(String dirname, NeuralNetworkManager neuralNetworkManager) {
		this.dirname = dirname;
		this.neuralNetworkManager = neuralNetworkManager;
	}
	
	public void test(){
        File folder = new File(dirname);
        File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".bmp"));

        if (files != null) {
            for (File file : files) {
                try {
                	Matrix input = getInputData(file);
                    int number = neuralNetworkManager.getPredictedNumber(input);
                    String parentPath = file.getParent();
                    String oldName = file.getName();
                    File newFile = new File(parentPath, number+ "_" + oldName);
                    file.renameTo(newFile);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
	
	public Matrix getInputData(File file) throws IOException {
        ImageBatchData batchData = new ImageBatchData();
		
    	BufferedImage image = ImageIO.read(file);

        int width = image.getWidth();
        int height = image.getHeight();
        
        int[] pixels = new int[width * height];

        image.getRGB(0, 0, width, height, pixels, 0, width);
        double[] data = new double[width * height];
        for (int i = 0; i < width * height; i++) {
        	int pixel = pixels[i];
        	int red = (pixel >> 16) & 0xFF;
            data[i] = red / 256.0;
        }
        batchData.setInputBatch(data);
        
        return new Matrix(width * height, 1, batchData.getInputBatch());
	}
}
