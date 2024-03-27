use linfa::dataset::DatasetBase;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use csv::{ReaderBuilder, Trim};
use std::error::Error;
use std::fs::File;
use std::fs;
use std::path::PathBuf;
use linfa::prelude::*;

// Function to read CSV file and parse it into ndarray arrays
fn read_csv(file_path: PathBuf) -> Result<(Array2<f64>, Array1<usize>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .trim(Trim::All)
        .from_reader(file);

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    // Iterate over CSV records and parse them
    for result in rdr.records() {
        let record = result?;
        let record_data: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        let target = record_data.last().cloned().unwrap() as usize;
        let record_without_target: Vec<f64> = record_data.iter().take(record_data.len() - 1).cloned().collect();
        records.push(record_without_target);
        targets.push(target);
    }

    // Convert parsed data into ndarray arrays
    let num_records = records.len();
    let num_features = if let Some(record) = records.first() { record.len() } else { 0 };
    let mut records_array = Array2::zeros((num_records, num_features));
    for (i, record) in records.iter().enumerate() {
        for (j, &value) in record.iter().enumerate() {
            records_array[[i, j]] = value;
        }
    }
    let targets_array = Array1::from(targets);

    Ok((records_array, targets_array))
}

// Function to split dataset into train and test sets
fn split_dataset(dataset: DatasetBase<Array2<f64>, Array1<usize>>, ratio: f64) -> (DatasetBase<Array2<f64>, Array1<usize>>, DatasetBase<Array2<f64>, Array1<usize>>) {
    let mut rng = rand::thread_rng();
    let n_samples = dataset.records().nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let n_train = (n_samples as f64 * ratio) as usize;
    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    // Split dataset into train and test sets based on shuffled indices
    let train_records = dataset.records().select(ndarray::Axis(0), train_indices).to_owned();
    let train_targets = dataset.targets().select(ndarray::Axis(0), train_indices).to_owned();
    let test_records = dataset.records().select(ndarray::Axis(0), test_indices).to_owned();
    let test_targets = dataset.targets().select(ndarray::Axis(0), test_indices).to_owned();

    let train_dataset = DatasetBase::new(train_records, train_targets);
    let test_dataset = DatasetBase::new(test_records, test_targets);

    (train_dataset, test_dataset)
}

// Function to calculate F1-score
fn f1_score(predictions: &Array1<usize>, targets: &[usize]) -> f64 {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    // Calculate true positives, false positives, and false negatives
    for (&predicted, &target) in predictions.iter().zip(targets.iter()) {
        if predicted == target {
            true_positives += 1;
        } else {
            if predicted == 1 && target == 0 {
                false_positives += 1;
            } else {
                false_negatives += 1;
            }
        }
    }

    // Calculate precision, recall, and F1-score
    let precision = true_positives as f64 / (true_positives + false_positives) as f64;
    let recall = true_positives as f64 / (true_positives + false_negatives) as f64;
    if precision + recall == 0.0 {
        return 0.0; // Handle division by zero
    }
    let f1 = 2.0 * (precision * recall) / (precision + recall);
    f1
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Decision Tree Classifier");

    // Path to the folder containing CSV files
    let folder_path = "../../_datasets_classification";

    // Read files in the folder
    let entries = fs::read_dir(folder_path)?;
    for entry in entries {
        println!("---------------------------------");
        let entry = entry?;
        let file_path = entry.path();
        if let Some(extension) = file_path.extension() {
            if extension == "csv" {
                // Load and process CSV file
                let file_name = file_path.file_name().unwrap().to_string_lossy();
                println!("Loading file: {}", file_name);
                let (records, targets) = read_csv(file_path)?;
                let dataset = DatasetBase::new(records, targets);

                // Split dataset into train and test sets
                let (train_set, test_set) = split_dataset(dataset, 0.8);

                // Train decision tree model
                let model = DecisionTree::params()
                    .split_quality(SplitQuality::Entropy)
                    .max_depth(Some(100))
                    .min_weight_split(1.0)
                    .min_weight_leaf(1.0)
                    .fit(&train_set)?;

                // Make predictions on test set
                let predictions = model.predict(&test_set);

                // Convert test targets to Vec<usize>
                let test_targets = test_set.targets().iter().map(|x| *x).collect::<Vec<usize>>();

                // Calculate F1-score
                let f1 = f1_score(&predictions, &test_targets);
                println!("F1-score: {}", f1);
            }
        }
    }

    Ok(())
}