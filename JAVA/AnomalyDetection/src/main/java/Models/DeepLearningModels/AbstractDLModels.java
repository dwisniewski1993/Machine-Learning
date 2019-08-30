package Models.DeepLearningModels;

import Iterators.AnomalyDetectionDataIterator;
import Models.AbstractModelInterface;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public abstract class AbstractDLModels implements AbstractModelInterface {
    private DataSetIterator normalDataIterator;
    private DataSetIterator attackDataIterator;
    private int numEpochs;
    private MultiLayerNetwork neuralNetwork;


    public AbstractDLModels(String normalDataPath, String attackDataPath, int batchSize, int numEpochs){
        this.normalDataIterator = new AnomalyDetectionDataIterator(normalDataPath, batchSize);
        this.attackDataIterator = new AnomalyDetectionDataIterator(attackDataPath, batchSize);
        this.numEpochs = numEpochs;
        NormalizeData();
    }

    public void NormalizeData(){
        DataNormalization normalization = new NormalizerMinMaxScaler();
        normalization.fit(this.normalDataIterator);
        this.normalDataIterator.reset();
        this.normalDataIterator.setPreProcessor(normalization);
        this.attackDataIterator.setPreProcessor(normalization);
    }

    public abstract void DefineNeuralNetwork(int numInput, int numOutput);

    public void TrainModel(){
        this.neuralNetwork.setListeners(new ScoreIterationListener());
        for (int i = 0; i < this.numEpochs; i++) {
            this.neuralNetwork.fit(this.normalDataIterator);
        }
    }

}
