package Models;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public interface AbstractModelInterface {
    void train() throws IOException;
    INDArray score(DataSetIterator dataSetIterator);
}
