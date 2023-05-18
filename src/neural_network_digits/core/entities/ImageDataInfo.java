package neural_network_digits.core.entities;

public class ImageDataInfo {
	private int imageWidth;
	private int imageHeight;

	private int numberItems;
	private int inputSize;
	private int expectedSize;
	private int numberBatches;
	private int totalItemsRead;
	private int itemsRead;

	public int getNumberItems() {
		return numberItems;
	}

	public void setNumberItems(int numberItems) {
		this.numberItems = numberItems;
	}

	public int getInputSize() {
		return inputSize;
	}

	public void setInputSize(int inputSize) {
		this.inputSize = inputSize;
	}

	public int getExpectedSize() {
		return expectedSize;
	}

	public void setExpectedSize(int expectedSize) {
		this.expectedSize = expectedSize;
	}

	public int getNumberBatches() {
		return numberBatches;
	}

	public void setNumberBatches(int numberBatches) {
		this.numberBatches = numberBatches;
	}

	public int getTotalItemsRead() {
		return totalItemsRead;
	}

	public void setTotalItemsRead(int totalItemsRead) {
		this.totalItemsRead = totalItemsRead;
	}

	public int getItemsRead() {
		return itemsRead;
	}
	
	public int getImageWidth() {
		return imageWidth;
	}

	public void setImageWidth(int imageWidth) {
		this.imageWidth = imageWidth;
	}

	public int getImageHeight() {
		return imageHeight;
	}

	public void setImageHeight(int height) {
		this.imageHeight = height;
	}


	public void setItemsRead(int itemsRead) {
		this.itemsRead = itemsRead;
		setTotalItemsRead(getTotalItemsRead() + itemsRead);
	}

}
