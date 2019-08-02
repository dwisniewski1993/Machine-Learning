import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.LinearRegression;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LinearRegressionModel {
    private static final Logger log = LoggerFactory.getLogger(LinearRegressionModel.class);
    private String arffFile;
    private int label;
    private Instances dataset;
    private Instances train;
    private Instances test;
    private LinearRegression model;

    LinearRegressionModel(String arffFilePath, int labelColumn){
        log.info("Initialize Linear Regression Model");
        this.arffFile = arffFilePath;
        this.label = labelColumn;
    }

    public void loadDataset() throws Exception {
        log.info("Loading dataset");
        DataSource source = new DataSource(this.arffFile);
        this.dataset = source.getDataSet();

        this.train = this.dataset.trainCV(80, 0);
        this.test = this.dataset.testCV(20, 0);

        this.train.setClassIndex(this.label);
        this.test.setClassIndex(this.label);
    }

    public void buildModel() throws Exception {
        log.info("Building model");
        this.model = new LinearRegression();
        this.model.buildClassifier(this.train);
    }

    public void predictAndEvaluate() throws Exception {
        int numAttr = this.test.numAttributes();
        int numInstances = this.test.numInstances();

        for (int instIdx = 0; instIdx < numInstances; instIdx++){
            Instance curInstance = this.test.instance(instIdx);

            double prediction = this.model.classifyInstance(curInstance);
            double actual = curInstance.toDoubleArray()[0];

            System.out.println("Predicted: " + prediction + ", actual: " + actual);
        }
    }

    public double predict() throws Exception {
        Instance myHouse = this.dataset.firstInstance();
        double price = this.model.classifyInstance(myHouse);

        return price;
    }

    public double getActual(){
        double[] actualHouse = this.dataset.firstInstance().toDoubleArray();
        return actualHouse[0];
    }

}
