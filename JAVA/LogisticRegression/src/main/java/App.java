import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws Exception {
        log.info("Logistic Regression with Weka");

        // Specify working directory and files names
        String currentDirectory = System.getProperty("user.dir");

        String csvFile = currentDirectory + "\\" +  "train.csv";
        String arffFile = currentDirectory + "\\" + "train.arff";

        boolean csvExist = Files.exists(Paths.get(csvFile));
        boolean arfExist = Files.exists(Paths.get(arffFile));

        if (csvExist && arfExist){
            log.info("Files detected");
        }
        else if (csvExist || !arfExist){
            log.info("Csv file detected");
            //Create Data Handler
            DataHandler dataHandler = new DataHandler(csvFile, arffFile);
            try {
                dataHandler.CsvToArff();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else {
            throw new FileNotFoundException("No files found - at least train csv file should be in project directory!");
        }

        // Build and run linear regression model
        LogisticRegressionModel clf = new LogisticRegressionModel(arffFile, 4);
        clf.loadDataset();
        clf.buildModel();
        clf.output();
    }
}
