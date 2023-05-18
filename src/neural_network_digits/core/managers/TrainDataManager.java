package neural_network_digits.core.managers;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import neural_network_digits.core.entities.ImageBatchData;
import neural_network_digits.core.entities.ImageDataInfo;

public class TrainDataManager {
    private String imageFileName;
    private String labelFileName;
    private int batchSize;

    private DataInputStream imageStream;
    private DataInputStream labelStream;

    private ImageDataInfo imageDataInfo;

    public TrainDataManager(String imageFileName, String labelFileName, int batchSize) {
        this.imageFileName = imageFileName;
        this.labelFileName = labelFileName;
        this.batchSize = batchSize;
    }

    public ImageDataInfo open() throws FileNotFoundException {
        imageStream = new DataInputStream(new FileInputStream(imageFileName));
        labelStream = new DataInputStream(new FileInputStream(labelFileName));
        return readImageDataInfo();
    }

    private ImageDataInfo readImageDataInfo() {
        imageDataInfo = new ImageDataInfo();

        try {
            int labelMagicNumber = labelStream.readInt();

            if (labelMagicNumber != 2049) {
                throw new IllegalArgumentException("Label file has wrong format.");
            }

            int itemCount = labelStream.readInt();
            imageDataInfo.setNumberItems(itemCount);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            int imageMagicNumber = imageStream.readInt();

            if (imageMagicNumber != 2051) {
                throw new IllegalArgumentException("Image file has wrong format.");
            }

            int itemCount = imageStream.readInt();

            if (itemCount != imageDataInfo.getNumberItems()) {
                throw new IllegalArgumentException("Image file has different number of items to label file.");
            }

            int height = imageStream.readInt();
            int width = imageStream.readInt();

            imageDataInfo.setImageHeight(height);
            imageDataInfo.setImageWidth(width);
            imageDataInfo.setInputSize(width * height);
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageDataInfo.setExpectedSize(10);
        imageDataInfo.setNumberBatches((int) Math.ceil((double) imageDataInfo.getNumberItems()) / batchSize);
        return imageDataInfo;
    }

    public void close() {
        try {
            imageStream.close();
            labelStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        imageDataInfo = null;
    }

    public ImageDataInfo getImageDataInfo() {
        return imageDataInfo;
    }

    public ImageBatchData readBatch() {
        ImageBatchData batchData = new ImageBatchData();

        int inputItemsRead = readInputBatch(batchData);
        int expectedItemsRead = readExpectedBatch(batchData);

        imageDataInfo.setItemsRead(inputItemsRead);

        return batchData;
    }

    private int readExpectedBatch(ImageBatchData batchData) {
        try {
            int totalItemsRead = imageDataInfo.getTotalItemsRead();
            int itemCount = imageDataInfo.getNumberItems();

            int itemsToRead = Math.min(itemCount - totalItemsRead, batchSize);

            byte[] labelData = new byte[itemsToRead];
            int expectedSize = imageDataInfo.getExpectedSize();

            int itemsRead = labelStream.read(labelData, 0, itemsToRead);

            if (itemsRead != itemsToRead) {
                throw new IllegalArgumentException("Couldn't read sufficient bytes from label data");
            }

            double[] data = new double[itemsToRead * expectedSize];
            for (int i = 0; i < itemsToRead; i++) {
                byte label = labelData[i];
                data[i * expectedSize + label] = 1;
            }

            batchData.setExpectedBatch(data);

            return itemsToRead;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0;
    }

    private int readInputBatch(ImageBatchData batchData) {
        try {
            int totalItemsRead = imageDataInfo.getTotalItemsRead();
            int itemCount = imageDataInfo.getNumberItems();

            int itemsToRead = Math.min(itemCount - totalItemsRead, batchSize);

            int inputSize = imageDataInfo.getInputSize();
            int bytesToRead = itemsToRead * inputSize;

            byte[] imageData = new byte[bytesToRead];

            int bytesRead = imageStream.read(imageData, 0, bytesToRead);

            double[] data = new double[bytesToRead];
            for (int i = 0; i < bytesToRead; i++) {
                data[i] = (imageData[i] & 0xFF) / 256.0;
            }

            batchData.setInputBatch(data);

            return itemsToRead;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0;
    }
}
