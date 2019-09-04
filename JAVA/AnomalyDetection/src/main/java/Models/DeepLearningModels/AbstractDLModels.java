package Models.DeepLearningModels;

import Iterators.AnomalyDetectionDataIterator;
import Models.AbstractModelInterface;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public abstract class AbstractDLModels implements AbstractModelInterface {
    private static final Logger log = LoggerFactory.getLogger(AbstractDLModels.class);
    private DataSetIterator normalDataIterator;
    private DataSetIterator attackDataIterator;
    private int numEpochs;
    public String modelName;
    public File modelPath;
    public MultiLayerNetwork neuralNetwork;

    public AbstractDLModels(String normalDataPath, String attackDataPath, int batchSize, int numEpochs){
        this.normalDataIterator = new AnomalyDetectionDataIterator(normalDataPath, batchSize);
        this.attackDataIterator = new AnomalyDetectionDataIterator(attackDataPath, batchSize);
        DefineNeuralNetwork(normalDataIterator.inputColumns(), normalDataIterator.totalOutcomes());
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

    public void train() throws IOException {
        log.info("Initializing " + this.modelName + " model...");
        if (this.modelPath.exists()){
            log.info("Loading detected model");
            loadModel();
        }
        else {
            log.info("Start training " + this.modelName + " model");
            this.neuralNetwork.setListeners(new ScoreIterationListener());
            for (int i = 0; i < this.numEpochs; i++) {
                this.neuralNetwork.fit(this.attackDataIterator);
            }
            saveModel();
        }
    }

    public void saveModel() throws IOException {
        ModelSerializer.writeModel(this.neuralNetwork, this.modelPath, false);
    }

    public void loadModel() throws IOException {
        this.neuralNetwork = ModelSerializer.restoreMultiLayerNetwork(this.modelPath);
    }

}
