package Models;

public class ResultsCalculator {
    private int truePositives;
    private int trueNegatives;
    private int falsePositives;
    private int falseNegatives;
    private double accuracy;
    private int totalPopulation;
    private int trueRatio;
    private int falseRatio;

    public ResultsCalculator(){}

    public double getAccuracy() {
        return this.accuracy;
    }

    private int getTotalPopulation(){
        return totalPopulation;
    }

    private int getTrueRatio(){
        return this.trueRatio;
    }

    public int getFalseRatio(){
        return this.trueRatio;
    }

    public int getFalseNegatives() {
        return falseNegatives;
    }

    public int getFalsePositives() {
        return falsePositives;
    }

    public int getTrueNegatives() {
        return trueNegatives;
    }

    public int getTruePositives() {
        return truePositives;
    }

    public void addTruePositive(){
        this.truePositives += 1;
        this.totalPopulation += 1;
        this.trueRatio += 1;
    }

    public void addFalsePositive(){
        this.falsePositives += 1;
        this.totalPopulation += 1;
        this.falseRatio += 1;
    }

    public void addTrueNegative(){
        this.trueNegatives += 1;
        this.totalPopulation += 1;
        this.trueRatio += 1;
    }

    public void addFalseNegative(){
        this.falseNegatives += 1;
        this.totalPopulation += 1;
        this.falseRatio += 1;
    }

    public void calculateAccuracy(){
        this.accuracy = (float)getTrueRatio() / getTotalPopulation();
    }
}
