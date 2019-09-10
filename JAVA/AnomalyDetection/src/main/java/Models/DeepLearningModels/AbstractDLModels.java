package Models.DeepLearningModels;

import Iterators.AnomalyDetectionDataIterator;
import Models.AbstractModelInterface;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class AbstractDLModels implements AbstractModelInterface {
    private static final Logger log = LoggerFactory.getLogger(AbstractDLModels.class);
    private AnomalyDetectionDataIterator normalDataIterator;
    private AnomalyDetectionDataIterator attackDataIterator;
    private int numEpochs;
    public String modelName;
    public File modelPath;
    public MultiLayerNetwork neuralNetwork;
    public double threshold;

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

    public AnomalyDetectionDataIterator getNormalDataIterator() {
        return normalDataIterator;
    }

    public AnomalyDetectionDataIterator getBrokenDataIterator(){
        return attackDataIterator;
    }

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

    public INDArray score(DataSetIterator data){
        log.info("Calculating " + this.modelName.replace("_", " ") + " model score");
        return this.neuralNetwork.output(data);
    }

    public void saveModel() throws IOException {
        ModelSerializer.writeModel(this.neuralNetwork, this.modelPath, false);
    }

    public void loadModel() throws IOException {
        this.neuralNetwork = ModelSerializer.restoreMultiLayerNetwork(this.modelPath);
    }

    public void calculateThreshold(INDArray predictions){
        log.info("Calculating " + this.modelName.replace("_", " ") + " model threshold...");
        INDArray actual = getActual();
        List distances = new ArrayList<Float>();
        for (int i = 0; i < this.normalDataIterator.totalExamples(); i++){
            distances.add(actual.get(NDArrayIndex.point(i)).distance2(predictions.get(NDArrayIndex.point(i))));
        }
        this.threshold = (double) Collections.max(distances);
    }

    public INDArray getActual(){
        this.normalDataIterator.reset();
        INDArray actual = this.normalDataIterator.next().getFeatures();
        while (this.normalDataIterator.hasNext()){
            actual = Nd4j.concat(0, actual, this.normalDataIterator.next().getFeatures());
        }
        return actual;
    }

    public void anomalyScore(INDArray predictions){
        log.info("Calculating anomaly score...");
        List distances = new ArrayList<Float>();
    }

}
