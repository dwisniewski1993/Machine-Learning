import Models.DeepLearningModels.LSTMNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;


public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws IOException {
        log.info("Anomaly Detection");

        // Specify working directory and files names
        String currentDirectory = System.getProperty("user.dir");

        String normalDataFilePath = currentDirectory + "\\" + "data_normal_small.csv";
        String attkDataFilePath = currentDirectory + "\\" + "data_attk_small.csv";

        LSTMNetwork ff = new LSTMNetwork(normalDataFilePath, attkDataFilePath, 10, 100);
        ff.train();

    }
}
