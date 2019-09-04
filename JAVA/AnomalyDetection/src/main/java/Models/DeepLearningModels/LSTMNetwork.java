package Models.DeepLearningModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class LSTMNetwork extends AbstractDLModels {
    private static final Logger log = LoggerFactory.getLogger(LSTMNetwork.class);

    public LSTMNetwork(String normalDataPath, String attackDataPath, int batchSize, int numEpochs) {
        super(normalDataPath, attackDataPath, batchSize, numEpochs);
        this.modelName = "LSTM_NETWORK";
        this.modelPath = new File(System.getProperty("user.dir") + "\\" + this.modelName + "_MODEL.zip");
    }

    @Override
    public void score(DataSetIterator dataSetIterator) {

    }

    @Override
    public void DefineNeuralNetwork(int numInput, int numOutput) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam.Builder().learningRate(0.001).build())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder()
                        .name("layer1")
                        .nIn(numInput)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder().name("layer2")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer3")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer4")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer5")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer6")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer7")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer8")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer9")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new LSTM.Builder()
                        .name("layer10")
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new RnnOutputLayer.Builder().name("output").nOut(numOutput)
                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        this.neuralNetwork = new MultiLayerNetwork(configuration);
        this.neuralNetwork.init();
    }
}
