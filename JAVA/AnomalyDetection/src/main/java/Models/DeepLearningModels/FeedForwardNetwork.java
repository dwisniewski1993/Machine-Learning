package Models.DeepLearningModels;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class FeedForwardNetwork extends AbstractDLModels {
    public FeedForwardNetwork(String normalDataPath, String attackDataPath, int batchSize, int numEpochs) {
        super(normalDataPath, attackDataPath, batchSize, numEpochs);
    }

    @Override
    public void train() {

    }

    @Override
    public void score(DataSetIterator dataSetIterator) {

    }

    @Override
    public void DefineNeuralNetwork(int numInput, int numOutput) {

    }
}
