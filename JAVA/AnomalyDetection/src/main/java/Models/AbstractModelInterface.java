package Models;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public interface AbstractModelInterface {
    void train() throws IOException;
    void score(DataSetIterator dataSetIterator);
}
