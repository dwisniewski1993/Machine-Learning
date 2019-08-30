package Models;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface AbstractModelInterface {
    public void train();
    public void score(DataSetIterator dataSetIterator);
}
