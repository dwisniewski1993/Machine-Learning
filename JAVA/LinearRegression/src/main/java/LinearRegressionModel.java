import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

class LinearRegressionModel {
    private static final Logger log = LoggerFactory.getLogger(LinearRegressionModel.class);
    private String arffFile;
    private int label;
    private Instances train;
    private Instances test;
    private LinearRegression model;

    LinearRegressionModel(String arffFilePath, int labelColumn){
        log.info("Initialize Linear Regression Model");
        this.arffFile = arffFilePath;
        this.label = labelColumn;
    }

    void loadDataset() throws Exception {
        log.info("Loading dataset");
        DataSource source = new DataSource(this.arffFile);
        Instances dataset = source.getDataSet();

        this.train = dataset.trainCV(80, 0);
        this.test = dataset.testCV(20, 0);

        this.train.setClassIndex(this.label);
        this.test.setClassIndex(this.label);
    }

    void buildModel() throws Exception {
        log.info("Building model");
        this.model = new LinearRegression();
        this.model.buildClassifier(this.train);
    }

    void output() throws Exception {
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        double mae = eval.rootMeanSquaredError();
        log.info("MSE: " + mae);
    }

}
