package Iterators;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class AnomalyDetectionDataReader {
    private int skipNumLines;
    private int skipNumColumns;
    private int longestTimeSequence;
    private int shortest;
    private Iterator<List<Writable>> iter;
    private Iterator<List<Writable>> labelIter;
    private Path filePath;
    private int totalExamples;
    private Queue<String> currentLines;

    public AnomalyDetectionDataReader(File file){
        this.skipNumLines = 1;
        this.skipNumColumns = 1;
        this.longestTimeSequence = 0;
        this.shortest = 1;
        this.filePath = file.toPath();
        this.currentLines = new LinkedList<String>();
        doInitialize();
    }

    public void doInitialize(){
        List<List<Writable>> dataLines = new ArrayList<>();
        List<List<Writable>> labelLines = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(filePath, Charset.forName("UTF-8"));
            for (int i = skipNumLines; i < lines.size(); i ++) {
                String tempStr = lines.get(i).replaceAll("\"", "")
                        .replaceAll(",",".");
                currentLines.offer(tempStr);
                int templength = tempStr.split(";").length - skipNumColumns;
                longestTimeSequence = longestTimeSequence < templength? templength:longestTimeSequence;
                List<Writable> dataLine = new ArrayList<>();
                List<Writable> labelLine = new ArrayList<>();
                String[] wary= tempStr.split(";");
                labelLine.add(new Text(wary[wary.length-1]));
                for (int j = skipNumColumns; j < wary.length-1; j++ ) {
                    dataLine.add(new Text(wary[j]));
                }
                dataLines.add(dataLine);
                labelLines.add(labelLine);
            }
        } catch (Exception e) {
            throw new RuntimeException("loading data failed");
        }
        iter = dataLines.iterator();
        labelIter = labelLines.iterator();
        totalExamples = dataLines.size();
    }

    public DataSet next(int num) {

        INDArray features = Nd4j.create(new int[]{num, shortest, longestTimeSequence}, 'f');
        INDArray featuresMask = Nd4j.ones(num, longestTimeSequence);
        for (int i = 0, k = 0; i < num && iter.hasNext(); i ++) {
            List<Writable> line= iter.next();
            int index = 0;
            for (Writable w: line) {
                features.putScalar(new int[]{i, k, index}, w.toDouble());
                ++index;
            }
            if (line.size() < longestTimeSequence) {// the default alignmentMode is ALIGN_START
                for(int step = line.size(); step < longestTimeSequence; step++) {
                    featuresMask.putScalar(i, step, 0.0D);
                }
            }
        }
        return new DataSet(features, features, featuresMask, featuresMask);
    }

    public boolean hasNext() {
        return iter != null && iter.hasNext();
    }

    public List<String> getLabels() {
        return null;
    }

    public void reset() {
        doInitialize();
    }

    public int totalExamples() {
        return totalExamples;
    }

    public Iterator<List<Writable>> getLabelIterator(){
        return this.labelIter;
    }

    public Queue<String> currentLines() {
        return currentLines;
    }

}
