import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

class DataHandler {
    private static final Logger log = LoggerFactory.getLogger(DataHandler.class);
    private String csvPath;
    private String arffPath;

    DataHandler(String csvFilePath, String arffFilePAth){
        this.csvPath = csvFilePath;
        this.arffPath = arffFilePAth;
        try {
            CsvToArff();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    void CsvToArff() throws Exception {
        log.info("Start converting file");

        // load the CSV file
        CSVLoader loader = new CSVLoader();

        loader.setSource(new File(this.csvPath));

        String [] options = new String[1];
        options[0] = "-H";
        loader.setOptions(options);

        Instances data = loader.getDataSet();

        //save as an ARFF
        ArffSaver saver = new ArffSaver();

        saver.setInstances(data);
        saver.setFile(new File(this.arffPath));
        saver.writeBatch();
    }
}
