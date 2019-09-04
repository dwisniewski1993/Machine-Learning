package Iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.List;
import java.util.Queue;

public class AnomalyDetectionDataIterator implements DataSetIterator {
    private AnomalyDetectionDataReader recordReader;
    private int batchSize;
    private DataSet last;
    private boolean useCurrent;
    private DataSetPreProcessor preProcessor;

    public AnomalyDetectionDataIterator(String filePath, int batchSize){
        this.recordReader = new AnomalyDetectionDataReader(new File(filePath));
        this.batchSize = batchSize;
    }

    @Override
    public DataSet next(int num) {
        DataSet ds = recordReader.next(num);
        if (preProcessor!=null){
            preProcessor.preProcess(ds);
        }
        return ds;
    }

    @Override
    public int inputColumns() {
        if (last == null){
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        }
        else {
            return last.numInputs();
        }
    }

    @Override
    public int totalOutcomes() {
        if (last == null){
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        }
        else {
            return last.numOutcomes();
        }
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        recordReader.reset();
        last = null;
        useCurrent = false;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public DataSet next() {
        if (useCurrent){
            useCurrent = false;
            return last;
        }
        else {
            return next(batchSize);
        }
    }

    public Queue<String> getCurrentLines(){
        return recordReader.currentLines();
    }
}
