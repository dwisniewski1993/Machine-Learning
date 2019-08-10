import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

class LogisticRegressionModel {
    private static final Logger log = LoggerFactory.getLogger(LogisticRegressionModel.class);
    private String arffFile;
    private int label;
    private Instances train;
    private Instances test;
    private Logistic model;

    LogisticRegressionModel(String arffFilePath, int labelColumn){
        log.info("Initialize Linear Regression Model");
        this.arffFile = arffFilePath;
        this.label = labelColumn;
    }

    void loadDataset() throws Exception {
        log.info("Loading dataset");
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(this.arffFile);
        Instances dataset = source.getDataSet();
        Random rand = new Random();
        dataset.randomize(rand);

        this.train = dataset.trainCV(80, 0);
        this.test = dataset.testCV(20, 0);

        this.train.setClassIndex(this.label);
        this.test.setClassIndex(this.label);
    }

    void buildModel() throws Exception {
        log.info("Building model");
        this.model = new Logistic();
        this.model.buildClassifier(this.train);
    }
    void output() throws Exception {
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        for (int i = 0; i < this.train.numClasses(); i++){
            double f1score = eval.fMeasure(i);
            log.info("F1 Score for index: " + i + " " + f1score);
        }
    }
}
