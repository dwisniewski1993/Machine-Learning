import Models.DeepLearningModels.FeedForwardNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args){
        log.info("Anomaly Detection");

        // Specify working directory and files names
        String currentDirectory = System.getProperty("user.dir");

        String normalDataFilePath = currentDirectory + "\\" + "data_normal_small.csv";
        String attkDataFilePath = currentDirectory + "\\" + "data_attk_small.csv";

        System.out.println(normalDataFilePath);
        System.out.println(attkDataFilePath);

        FeedForwardNetwork ff = new FeedForwardNetwork(normalDataFilePath, attkDataFilePath, 64, 100);

    }
}
